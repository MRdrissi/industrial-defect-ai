# EPIC-05 — Évaluation scientifique et benchmarking

**Branche commune** — `epic/05-evaluation-benchmark`

L'Epic compare CAE et PatchCore sur Bottle, Hazelnut et Transistor avec les splits figés de l'EPIC-02.

| User Story | Périmètre | État |
|---|---|---|
| [US-05-01 — Pipeline commun](epic-05/us-05-01.md) | Métriques image/pixel, masques, latence, exports et protocole | Tests synthétiques validés ; revue collaborative restante |
| [US-05-02](epic-05/us-05-02.md) | Calibration des six seuils sur Validation normale | 6/6 calibrations réelles archivées ; registre strict exporté, rechargé et validé sans nouvelle inférence |
| [US-05-03](epic-05/us-05-03.md) | Benchmark Test et macro-moyennes | Phase 1 : orchestrateur et tests synthétiques livrés ; exécution réelle et macro-moyennes non réalisées |
| US-05-04 | Faux positifs, faux négatifs et limites scientifiques | Hors de la présente livraison |

Le [protocole commun](../experiments/configs/evaluation_protocol.json) distingue calibration sur Validation
et évaluation sur Test. Aucun résultat de détection réel n'est produit par US-05-01.

US-05-02 fournit le [registre des six seuils](../experiments/calibration/six-thresholds/thresholds.json),
avec son ancre d'intégrité dans [availability.json](../experiments/calibration/availability.json).
Les sessions Kaggle 369724364 et 369945024 sont archivées séparément et conservées octet pour octet.
US-05-03 doit charger ces seuils figés, sans les ajuster sur Test.
