"""
Module de réassociation par apparence (Re-ID) pour le tracking hybride.

Maintient une « galerie » d'embeddings visuels des IDs récemment perdus.
Quand ByteTrack attribue un **nouvel** ID à une personne réapparue après
une occlusion prolongée, ce module compare son apparence à la galerie et
remappe l'ID vers l'ancien si la similarité cosinus dépasse le seuil.

Le calcul d'embedding n'est déclenché que pour les IDs effectivement en
occlusion prolongée ou nouvellement apparus, pas à chaque frame pour tous
les IDs, afin de préserver les performances temps réel.
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class ReIDGallery:
    """
    Galerie de réassociation d'identité par apparence (embedding cosinus).

    Attributes:
        occlusion_threshold (int): Nombre de frames d'absence consécutives
            avant qu'un ID soit archivé dans la galerie.
        gallery_ttl (int): Durée de vie maximale (en frames) d'une entrée
            dans la galerie avant purge automatique.
        similarity_threshold (float): Seuil de similarité cosinus pour
            considérer qu'un nouvel ID correspond à un ancien.
        id_remap (Dict[int, int]): Table de correspondance nouvel_ID → ancien_ID.
    """

    def __init__(
        self,
        occlusion_threshold: int = 1,
        gallery_ttl: int = 180,
        similarity_threshold: float = 0.45,
    ):
        """
        Initialise la galerie Re-ID.

        Args:
            occlusion_threshold: Frames d'absence avant archivage en galerie (défaut 1 pour archivage immédiat).
            gallery_ttl: Durée de vie max d'une entrée en galerie en frames (défaut 180, soit ~6 secondes).
            similarity_threshold: Seuil cosinus pour le remapping (défaut 0.45 pour haute réceptivité).
        """
        self.occlusion_threshold = occlusion_threshold
        self.gallery_ttl = gallery_ttl
        self.similarity_threshold = similarity_threshold

        # Table de remapping : new_id → old_id
        self.id_remap: Dict[int, int] = {}

        # Galerie des IDs perdus : old_id → {embedding, frame_lost}
        self._gallery: Dict[int, dict] = {}

        # Compteurs d'absence par ID (frames consécutives sans détection)
        self._absence_counter: Dict[int, int] = {}

        # Derniers embeddings connus pour les IDs actifs
        self._active_embeddings: Dict[int, np.ndarray] = {}

        # Ensemble des IDs vus à la frame précédente
        self._prev_active_ids: Set[int] = set()

        # Ensemble de tous les IDs déjà rencontrés (pour détecter les « nouveaux »)
        self._known_ids: Set[int] = set()

        # Compteur de frames global
        self._frame_count: int = 0

        # Réseau d'extraction d'embeddings (chargement paresseux)
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def remap_ids(
        self,
        track_ids: List[int],
        xyxy_list: np.ndarray,
        frame: np.ndarray,
        tracker_states: Optional[Dict] = None,
    ) -> List[int]:
        """
        Applique le remapping d'IDs en comparant les nouveaux IDs à la galerie.
        """
        self._frame_count += 1
        current_ids = set(track_ids)

        # --- 1. Détection des IDs disparus → archivage immédiat en galerie ---
        disappeared_ids = self._prev_active_ids - current_ids
        for did in disappeared_ids:
            real_id = self._resolve_real_id(did)
            self._absence_counter[real_id] = self._absence_counter.get(real_id, 0) + 1

            # Archiver en galerie dès la disparition
            if real_id in self._active_embeddings:
                self._gallery[real_id] = {
                    "embedding": self._active_embeddings[real_id],
                    "frame_archived": self._frame_count,
                }

        # --- 2. Identifier les IDs nouvellement apparus ---
        new_ids_in_frame: List[int] = []
        for tid in track_ids:
            if tid not in self._known_ids:
                new_ids_in_frame.append(tid)
                self._known_ids.add(tid)

        # --- 3. Tenter le remapping pour chaque nouvel ID ---
        remapped = list(track_ids)
        for i, tid in enumerate(track_ids):
            if tid in new_ids_in_frame and len(self._gallery) > 0:
                bbox = xyxy_list[i]
                new_emb = self._extract_embedding(frame, bbox)
                if new_emb is not None:
                    best_match_id, best_sim = self._find_best_match(new_emb)
                    if best_match_id is not None:
                        # Remapper le nouvel ID vers l'ancien ID conservé
                        self.id_remap[tid] = best_match_id
                        remapped[i] = best_match_id
                        self._active_embeddings[best_match_id] = new_emb
                        self._absence_counter.pop(best_match_id, None)
                        self._gallery.pop(best_match_id, None)
                        print(
                            f"[Re-ID] Identité conservée ! ID #{tid} → réassocié à l'ancien ID #{best_match_id} "
                            f"(similarité = {best_sim:.3f})"
                        )
                        continue

            # Appliquer le remapping existant si déjà connu
            real_id = self.id_remap.get(tid, tid)
            remapped[i] = real_id

            # Mettre à jour l'embedding actif régulièrement (toutes les 3 frames ou à l'apparition)
            if self._frame_count % 3 == 0 or tid in new_ids_in_frame:
                bbox = xyxy_list[i]
                emb = self._extract_embedding(frame, bbox)
                if emb is not None:
                    self._active_embeddings[real_id] = emb

            # Réinitialiser le compteur d'absence
            self._absence_counter[real_id] = 0

        # --- 4. Purge de la galerie (entrées trop anciennes) ---
        expired = [
            gid
            for gid, info in self._gallery.items()
            if (self._frame_count - info["frame_archived"]) > self.gallery_ttl
        ]
        for gid in expired:
            self._gallery.pop(gid, None)

        self._prev_active_ids = set(remapped)
        return remapped

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _resolve_real_id(self, tid: int) -> int:
        """Résout le remapping inverse pour obtenir l'ID réel."""
        # Chercher dans id_remap les valeurs (ancien_id)
        for new_id, old_id in self.id_remap.items():
            if new_id == tid:
                return old_id
        return tid

    def _find_best_match(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Trouve la meilleure correspondance dans la galerie par similarité cosinus.

        Returns:
            (best_id, best_similarity) ou (None, 0.0) si aucune correspondance.
        """
        best_id: Optional[int] = None
        best_sim: float = 0.0

        for gid, info in self._gallery.items():
            sim = self._cosine_similarity(embedding, info["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        return None, 0.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux vecteurs."""
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(dot / (na * nb))

    def _ensure_model_loaded(self) -> None:
        """Charge le réseau MobileNetV3 Small à la première utilisation (lazy)."""
        if self._model is not None:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Re-ID] Chargement du réseau Re-ID (MobileNetV3 Small) sur {self._device}...")

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Remplacer le classifieur par Identity pour obtenir le vecteur d'embedding
        backbone.classifier = nn.Identity()
        backbone.eval()
        self._model = backbone.to(self._device)

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def _extract_embedding(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extrait un vecteur d'embedding L2-normalisé à partir du crop d'une bbox.

        Args:
            frame: Image BGR OpenCV.
            bbox: [x1, y1, x2, y2].

        Returns:
            Vecteur NumPy L2-normalisé, ou None si le crop est invalide.
        """
        self._ensure_model_loaded()

        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self._transform(crop_rgb).unsqueeze(0).to(self._device)
        embedding = self._model(tensor).squeeze(0).cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding
