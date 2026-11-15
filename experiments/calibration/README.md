# Artefacts de calibration — US-05-02

[Documentation et commandes](../../docs/epic-05/us-05-02.md) ·
[Catalogue des six instances](../configs/calibration_instances.json) ·
[État des Inputs vérifié](availability.json)

**Quatre calibrations réelles sur six sont archivées** dans [kaggle-001/](kaggle-001/).
CAE Bottle, CAE Hazelnut, PatchCore Hazelnut EXP-002 et PatchCore Transistor disposent de scores
Validation et de seuils rechargeables sans recalcul. CAE Transistor et PatchCore Bottle restent
non calibrés ; aucun registre complet n'est créé.

Les 13 fichiers originaux sont conservés octet pour octet. La [provenance séparée](kaggle-001-provenance.json)
enregistre la version Kaggle 369724364, le commit scientifique c708de9 et les empreintes d'intégrité.
Ne pas utiliser les valeurs synthétiques des tests comme des résultats scientifiques.

Après exécution, conserver un dossier neuf par instance :
`validation_scores.csv`, `calibration.json`, `integrity.json`.
Le registre `six-thresholds/` ne doit être créé qu'après six calibrations complètes.
Le SHA-256 d'intégrité retourné par l'exporteur doit être conservé dans une référence externe revue.

Les archives historiques, checkpoints et datasets restent à leur emplacement d'origine.
