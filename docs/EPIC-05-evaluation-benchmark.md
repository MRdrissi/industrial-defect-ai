# EPIC-05 — Évaluation scientifique et benchmarking

**Branche commune** — `epic/05-evaluation-benchmark`

L'Epic compare CAE et PatchCore sur Bottle, Hazelnut et Transistor avec les splits figés de l'EPIC-02.

| User Story | Périmètre | État |
|---|---|---|
| [US-05-01 — Pipeline commun](epic-05/us-05-01.md) | Métriques image/pixel, masques, latence, exports et protocole | Tests synthétiques validés ; revue collaborative restante |
| [US-05-02](epic-05/us-05-02.md) | Calibration des six seuils sur Validation normale | 6/6 calibrations réelles archivées ; registre strict exporté, rechargé et validé sans nouvelle inférence |
| [US-05-03](epic-05/us-05-03.md) | Benchmark Test et macro-moyennes | Livraison technique complète **6/6** ; 586 prédictions/NPZ audités, comparaison et macros sur trois catégories ; revue collaborative distincte |
| [US-05-04](epic-05/us-05-04.md) | Faux positifs, faux négatifs et limites scientifiques | Clôturable techniquement : 86 erreurs inventoriées, 13 figures comparatives et trois critères IDL satisfaits ; revue humaine distincte |

Le [protocole commun](../experiments/configs/evaluation_protocol.json) distingue calibration sur Validation
et évaluation sur Test. US-05-02 fournit le [registre des six seuils](../experiments/calibration/six-thresholds/thresholds.json),
avec son ancre d'intégrité dans [availability.json](../experiments/calibration/availability.json).
Les sessions de calibration Kaggle 369724364 et 369945024 restent inchangées.

US-05-03 réutilise ces seuils sans ajustement sur Test. Les sorties réelles des versions Kaggle
**370020172** (2 instances) et **370053025** (4 instances) sont archivées séparément,
octet pour octet, avec [provenance et politique de conservation](../experiments/benchmark/README.md).
Les 586 NPZ restent hors Git, vérifiés dans les deux ZIP originaux. Les deux familles sont comparées
sur les mêmes **293 images Test distinctes**, sans nouvelle exécution des modèles.

Les [macro-moyennes](../experiments/benchmark/summary.json) donnent un poids de 1/3 à chaque catégorie :
F1 **0,617776 CAE / 0,969924 PatchCore**, Image-AUROC **0,807553 / 0,996944**,
Pixel-AUROC **0,810604 / 0,976739**, latence moyenne **19,569520 / 28,159651 ms**.
Ce sont des résultats descriptifs à une exécution par instance sur Tesla T4, issus de deux sessions,
sans mesure de variance entre exécutions ni garantie hors de ce périmètre.

Validation : **296 tests et 2 sous-tests réussis**, dont 39 contrôles d'artefacts/synthèse.
La [fiche US-05-03](epic-05/us-05-03.md) fournit les six résultats, seuils, latences,
formules, limites et bilan DoD. Aucun statut Jira, aucune PR ni aucun merge n'est réalisé ici.

US-05-04 complète l'évaluation par une [analyse des erreurs](epic-05/us-05-04.md) et une
[galerie de 13 exemples](epic-05/us-05-04-gallery.md) couvrant les trois catégories et toutes les
strates d'erreurs observées. Les deux modèles sont affichés sur les mêmes RGB et GT ; 24 fichiers
sources Kaggle v2 ont été récupérés et vérifiés. Aucun cas d'erreur n'est inventé pour PatchCore
Hazelnut. L'analyse distingue classification image et localisation, documente les limites des seuils
et propose PatchCore pour le prototype lorsque le rappel prime, avec son surcoût de latence et ses FP.
Validation étendue US-05-04 : **323 tests et 2 sous-tests**, dont 27 nouveaux ; figures reproduites
octet pour octet. Les six benchmarks, seuils et notebook EDA restent inchangés.
