# EPIC-06 — Inference & Web Application

**Branche :** `epic/06-inference-web-app`

## Objectif

Développer une application Streamlit pour l'inspection visuelle,
en séparant l'interface utilisateur des moteurs d'inférence.

## Organisation

| US | Objectif | Documentation |
|---|---|---|
| US-06-01 | Contrat commun, registre et simulation CPU | [US-06-01](epic-06/us-06-01.md) |
| US-06-02 | Interface Streamlit et importation d'image | [US-06-02](epic-06/us-06-02.md) |
| US-06-03 | Intégration des six modèles réels | [US-06-03](epic-06/us-06-03.md) |
| US-06-04 | Heatmap superposée, décision, version et rapport JSON | [US-06-04](epic-06/us-06-04.md) |
| US-06-05 | Comparaison CAE / PatchCore sur une même image | [US-06-05](epic-06/us-06-05.md) |

## Architecture

- `src/inference/` : contrat et moteurs d'inférence.
- `configs/model_registry.json` : registre.
- `app/streamlit_app.py` : interface Streamlit et orchestration de l'inférence.
- `app/image_utils.py` : validation, chargement et conversion des images.
- `app/result_view.py` : affichage des résultats, scores et cartes d'anomalie.
- `app/requirements.txt` : dépendances de l'application web.
- `tests/inference/` : tests automatisés.
- `configs/real_model_registry.json` : registre des six modèles réels.
- `src/inference/model_loader.py` : restauration CPU, vérification SHA-256 et cache.
- `src/inference/cae_inference.py` : inférence CAE.
- `src/inference/patchcore_inference.py` : inférence PatchCore.
- `app/overlay.py` : superposition de la carte d'anomalie sur l'image (US-06-04).
- `app/report.py` : rapport d'inspection JSON traçable (US-06-04).
- `app/comparison.py`, `app/comparison_view.py` : comparaison CAE / PatchCore côte à côte (US-06-05).

## Principes

- Les modèles sont identifiés par catégorie, méthode et version.
- Le contrat commun contient score, seuil, carte, décision et latence.
- La simulation est explicitement distinguée de l'inférence réelle.
- Les seuils scientifiques seront intégrés après calibration EPIC-05.

## Intégration réelle

Les six modèles CAE/PatchCore des catégories Bottle, Hazelnut
et Transistor sont intégrés et exécutables sur CPU (US-06-03).

L'application Streamlit propose désormais deux modes :
simulation et inférence réelle.

Les checkpoints sont restaurés à la demande, avec vérification
SHA-256 et cache limité à une instance.

La décision reste non calibrée tant que les seuils de
l'EPIC-05 ne sont pas intégrés.
