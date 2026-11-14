# EPIC-06 — Inference & Web Application

**Branche :** `epic/06-inference-web-app`

## Objectif

Développer une application Streamlit pour l'inspection visuelle,
en séparant l'interface utilisateur des moteurs d'inférence.

## Organisation

| US | Objectif | Documentation |
|---|---|---|
| US-06-01 | Contrat commun, registre et simulation CPU | [US-06-01](epic-06/us-06-01.md) |
| US-06-02 | Interface Streamlit et importation d'image | À venir |
| US-06-03 | Intégration des six modèles réels | À venir |

## Architecture

- `src/inference/` : contrat et moteurs d'inférence.
- `configs/model_registry.json` : registre.
- `app/` : future application Streamlit.
- `tests/inference/` : tests automatisés.

## Principes

- Les modèles sont identifiés par catégorie, méthode et version.
- Le contrat commun contient score, seuil, carte, décision et latence.
- La simulation est explicitement distinguée de l'inférence réelle.
- Les seuils scientifiques seront intégrés après calibration EPIC-05.
- Les checkpoints réels seront connectés dans US-06-03.
