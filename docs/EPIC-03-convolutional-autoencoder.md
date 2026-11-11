# EPIC-03 — Convolutional Autoencoder

**Branche commune** — `epic/03-convolutional-autoencoder`

Cet Epic utilise les manifestes et le profil de prétraitement CAE livrés dans [EPIC-02](EPIC-02-data-preparation.md). Les entraînements par catégorie partent d’une même architecture de reconstruction RGB.

| User Story | Livrable | Validation |
|---|---|---|
| [US-03-01 — Architecture du CAE](epic-03/us-03-01.md) | Modèle PyTorch configurable et configuration de référence | Dimensions de reconstruction, MSE, rétropropagation et mise à jour des poids sur CPU |
| [US-03-03 — CAE Transistor](epic-03/us-03-03.md) | Checkpoint `transistor` (cae-transistor/EXP-001), notebooks d'entraînement et de restauration | Meilleure MSE val 0,000612 (époque 98) ; checkpoint rechargé sur CPU avec la même perte |
| [US-03-04 — CAE Hazelnut](epic-03/us-03-04.md) | Entraînement et checkpoint [`cae-hazelnut/EXP-001`](../experiments/runs/cae-hazelnut/EXP-001.json) | Minimum de MSE à l’époque 147, restauration CPU sur 78 images normales |

La fiche US-03-01 décrit l’architecture et ses tests. La fiche Hazelnut documente le protocole, les résultats et la provenance des poids. La comparaison des capacités du bottleneck et l’évaluation des cartes d’anomalies relèvent des travaux ultérieurs.
