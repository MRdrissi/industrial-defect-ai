# EPIC-04 — PatchCore

**Branche commune** — `epic/04-patchcore`

PatchCore exploite les images normales des [splits EPIC-02](epic-02/us-02-03.md) et le [pipeline commun](epic-02/us-02-04.md) pour constituer une banque de caractéristiques locales.

| User Story | Livrable | Validation |
|---|---|---|
| [US-04-01 — Intégration Kaggle](epic-04/us-04-01.md) | Smoke test Bottle et restauration du checkpoint | 819 patches mémorisés, inférence sur une normale de validation, comparaison GPU/CPU dans les tolérances |
| [US-04-02 — Memory Bank Bottle](epic-04/us-04-02.md) | Memory Bank complète sur 167 images normales | 1 710 patches conservés, checkpoint sauvegardé et restauration CPU validée |
| [US-04-04 — Memory Bank Hazelnut](epic-04/us-04-04.md) | EXP-001 archivée : 313 normales, 3 205 vecteurs, rapports GPU/CPU | Restauration PASS rapportée ; protocole non validé (pooling différent entre construction et inférence). Ne pas utiliser pour US-05-02. |
