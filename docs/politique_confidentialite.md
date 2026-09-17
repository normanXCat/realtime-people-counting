# Politique de confidentialité — vidéos annotées et journaux d'événements

Document court, à joindre au mémoire. Il couvre uniquement les données produites
par le moteur vidéo (vidéos annotées, journaux d'événements, configurations
résolues). L'interface web est hors périmètre.

## 1. Nature des données traitées

Le système traite des images de personnes dans un espace **privé ou
semi-privé** (salle, couloir, entrée de bâtiment) et produit :

- `annotated.mp4` : vidéo avec boîtes englobantes, ancres et ligne virtuelle ;
- `events.jsonl` : journal JSON Lines (ligne validée, franchissements,
  occupations, événements ReID, purges, incohérences) ;
- `calibration.json` : ligne virtuelle validée par l'opérateur (deux points
  normalisés, clics bruts dans la fenêtre, côté intérieur, résolution) — aucune
  donnée personnelle ;
- `summary.json`, `environment.json`, `config_resolved.yaml` : traçabilité du run.

Aucune donnée biométrique **nominative** n'est produite : les identifiants
`person_id`, `technical_track_id` et `event_id` sont des identifiants de session,
sans lien avec l'état civil. Un descripteur d'apparence (histogramme couleur par
défaut, ou embedding profond si `reid.external_reid.enabled: true`) est conservé
en mémoire vive pour la réassociation, **jamais écrit dans le journal** et
**jamais exporté** : les événements `REID_*` ne contiennent que des scores de
similarité.

## 2. Finalité et base juridique

Les traitements servent uniquement à la mesure d'affluence (comptage
entrées/sorties, occupation) dans un but de recherche académique et de
statistique d'occupation. Aucune finalité de surveillance individuelle, de
profilage, de reconnaissance faciale ou de ré-identification inter-sites n'est
poursuivie.

## 3. Principes appliqués

- **Minimisation** : seules les boîtes de la classe *person* sont conservées ; le
  pipeline ne traite pas les visages (pas de modèle de reconnaissance faciale) ;
  les descripteurs d'apparence ne sont pas journalisés.
- **Durée de conservation** : les sessions sont conservées le temps de
  l'expérimentation et de la rédaction du mémoire, puis supprimées au plus tard
  **12 mois** après la soutenance. Les vidéos annotées sont supprimées en premier,
  les journaux agrégés peuvent être conservés à des fins de reproductibilité
  (voir §5).
- **Accès** : accès limité à l'auteur du mémoire et à son encadrant. Les
  résultats stockés sous `results/` ne sont ni poussés sur un dépôt public ni
  déposés sur un service tiers. Le dossier `results/` est exclu du dépôt Git
  (voir `.gitignore`).
- **Sécurité** : traitement en local (aucun envoi d'images vers un service
  externe), dépendances figées, poids vérifiés par empreinte SHA-256
  (`model.expected_sha256`).

## 4. Information des personnes

Pour un espace recevant du public, une signalétique indiquant la présence d'un
dispositif de comptage automatique (sans reconnaissance faciale, avec durée de
conservation) doit être affichée à l'entrée. Pour un espace privé, une
information écrite préalable des occupants est requise. Les vidéos de
recherche doivent être acquises avec l'accord préalable du responsable du lieu.

## 5. Anonymisation et partage des résultats

- Les **captures d'illustration** destinées au mémoire montrent des personnes de
  dos ou floutées ; à défaut, une autorisation écrite des personnes identifiables
  est nécessaire.
- Les **journaux d'événements** (sans image) peuvent être conservés plus
  longtemps : ils ne contiennent aucune donnée permettant de remonter à une
  personne, seulement des horodatages, directions et statistiques.
- Les **vidéos brutes** et les vidéos annotées ne sont jamais publiées.

## 6. Droits des personnes

Conformément à la réglementation applicable, toute personne concernée peut
demander l'accès, la rectification ou l'effacement des données la concernant,
ainsi que la limitation du traitement. La demande est adressée à l'auteur du
mémoire ; l'effacement d'un enregistrement peut nécessiter la reprise des
mesures concernées, ce qui est documenté dans le rapport final.

## 7. Registre des sessions

Chaque run écrit son `session_id` et la liste de ses sorties. Le registre tenu à
part indique, pour chaque session : le lieu, la date, la finalité, la durée de
conservation prévue et la date de suppression effective. C'est ce registre qui
rend la durée de conservation vérifiable en soutenance.
