# EPIC-05 — Évaluation scientifique et benchmarking

**Branche commune** — `epic/05-evaluation-benchmark`

L'Epic compare CAE et PatchCore sur Bottle, Hazelnut et Transistor avec les splits figés de l'EPIC-02.

| User Story | Périmètre | État |
|---|---|---|
| [US-05-01 — Pipeline commun](epic-05/us-05-01.md) | Métriques image/pixel, masques, latence, exports et protocole | Tests synthétiques validés ; revue collaborative restante |
| [US-05-02](epic-05/us-05-02.md) | Calibration des six seuils sur Validation normale | 4/6 calibrations réelles archivées ; CAE Transistor et PatchCore Bottle attendent leurs checkpoints |
| US-05-03 | Benchmark Test et macro-moyennes | Hors de la présente livraison |
| US-05-04 | Faux positifs, faux négatifs et limites scientifiques | Hors de la présente livraison |

Le [protocole commun](../experiments/configs/evaluation_protocol.json) distingue calibration sur Validation
et évaluation sur Test. Aucun résultat de détection réel n'est produit par US-05-01.
