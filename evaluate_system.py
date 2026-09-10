import json
import math
import os


def charger_evenements(chemin_fichier):
    """Charge un fichier JSON contenant la liste des événements."""
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(
            f"Le fichier spécifié est introuvable : {chemin_fichier}"
        )

    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluer_comptage_systeme(
    chemin_gt, chemin_pred, tolerance_frames=30
):
    """Compare la Vérité Terrain avec les prédictions et calcule les métriques de précision et de latence."""

    # 1. Chargement des événements
    gt_events = charger_evenements(chemin_gt)
    pred_events = charger_evenements(chemin_pred)

    # 2. Analyse statistique de la Latence (T_latency)
    latences = [
        e["latency_ms"] for e in pred_events if "latency_ms" in e
    ]

    if latences:
        latence_moyenne = sum(latences) / len(latences)
        latence_min = min(latences)
        latence_max = max(latences)
    else:
        latence_moyenne = latence_min = latence_max = 0.0

    # 3. Initialisation des métriques de franchissement
    vrais_positifs = 0  # TP
    faux_positifs = 0  # FP
    faux_negatifs = 0  # FN
    erreurs_direction = 0

    gt_restants = list(gt_events)

    # 4. Traitement et appariement temporel des événements
    for pred in pred_events:
        frame_p = pred["frame"]
        dir_p = pred["direction"]

        meilleur_match_idx = None
        distance_min_frame = math.inf

        for idx, gt in enumerate(gt_restants):
            frame_g = gt["frame"]
            ecart = abs(frame_p - frame_g)

            if ecart <= tolerance_frames and ecart < distance_min_frame:
                distance_min_frame = ecart
                meilleur_match_idx = idx

        if meilleur_match_idx is not None:
            gt_associe = gt_restants.pop(meilleur_match_idx)
            dir_g = gt_associe["direction"]

            if dir_p == dir_g:
                vrais_positifs += 1
            else:
                erreurs_direction += 1
                faux_positifs += 1
                faux_negatifs += 1
        else:
            faux_positifs += 1

    faux_negatifs += len(gt_restants)

    # 5. Calcul des métriques volumétriques
    total_gt = len(gt_events)
    total_pred = len(pred_events)

    precision = (
        vrais_positifs / (vrais_positifs + faux_positifs)
        if (vrais_positifs + faux_positifs) > 0
        else 0.0
    )
    recall = (
        vrais_positifs / (vrais_positifs + faux_negatifs)
        if (vrais_positifs + faux_negatifs) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy_volumetrique = (
        (1.0 - (abs(total_gt - total_pred) / total_gt)) * 100
        if total_gt > 0
        else 0.0
    )

    # 6. Affichage du rapport récapitulatif
    print("==================================================")
    print("        ÉVALUATION DU SYSTÈME GLOBAL DE COMPTAGE   ")
    print("==================================================")
    print(f"Total Vérité Terrain (GT) : {total_gt}")
    print(f"Total Prédit (Système)    : {total_pred}")
    print("--------------------------------------------------")
    print(f"Vrais Positifs (TP)       : {vrais_positifs}")
    print(f"Faux Positifs (FP)       : {faux_positifs}")
    print(f"Faux Négatifs (FN)       : {faux_negatifs}")
    print(f"Erreurs de Direction      : {erreurs_direction}")
    print("--------------------------------------------------")
    print(f"Précision (Precision)     : {precision * 100:.2f} %")
    print(f"Rappel (Recall)           : {recall * 100:.2f} %")
    print(f"F1-Score                  : {f1_score * 100:.2f} %")
    print(f"Exactitude Volumétrique   : {accuracy_volumetrique:.2f} %")
    print("--------------------------------------------------")
    print(" PERFORMANCES TEMPORELLES (LATENCE SYSTÈME)")
    print(f" Latence Moyenne           : {latence_moyenne:.2f} ms")
    print(
        f" Extrêmes (Min / Max)      : {latence_min:.2f} ms / {latence_max:.2f} ms"
    )
    print("==================================================")


if __name__ == "__main__":
    evaluer_comptage_systeme(
        chemin_gt="ground_truth.json",
        chemin_pred="predictions_systeme.json",
        tolerance_frames=30,
    )
