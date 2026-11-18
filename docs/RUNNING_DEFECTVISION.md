# Lancer DefectVision avec les six modèles réels

Environnement audité le 18 novembre 2026 : Windows, Python 3.12.6, CPU.
Les versions complètes sont figées dans `app/requirements-lock-windows-cpu.txt`.
Les poids ne sont pas dans Git. Un clone seul ne suffit pas.

## Installation

Depuis la racine du dépôt, dans PowerShell, créer un environnement uniquement
s'il n'en existe pas déjà :

```powershell
py -3.12 -m venv .venv
```

Installer les roues CPU depuis leur source officielle, puis les dépendances :

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu torchvision==0.29.2+cpu
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Le verrouillage vise Windows/Python 3.12 CPU. Aucun environnement Linux, GPU ou
hébergement Streamlit Cloud n'a été validé par cet audit. Anomalib est nécessaire
à PatchCore ; le seul paquet Streamlit ne suffit pas.

## Restaurer les artefacts

Le registre `configs/real_model_registry.json` définit les six destinations et
leurs empreintes SHA-256. Le catalogue `experiments/configs/calibration_instances.json`
fournit les expériences et sources Kaggle correspondantes.

Les sources locales retrouvées lors de l'audit permettent cette restauration :

```powershell
.\.venv\Scripts\python.exe scripts/restore_app_artifacts.py `
  --source "$env:USERPROFILE\Downloads\us0503-remaining-checkpoints.zip" `
  --source "$env:USERPROFILE\Downloads\us0502-kaggle-input\cae\transistor\best_model.pt" `
  --source "$env:USERPROFILE\Downloads\us0502-kaggle-input\patchcore\bottle\EXP-002\patchcore_model.pt" `
  --apply --report artifacts/restoration-report.json
```

Adapter les sources à l'emplacement de la sauvegarde. Sans `--apply`, le script
vérifie et présente le plan dans le rapport. Il identifie les fichiers par leur
SHA-256, lit les membres ZIP sans extraire leurs chemins et refuse tout écrasement
d'un fichier différent. Les sources restent intactes. Un hash inconnu ne remplace
jamais un modèle attendu. Pour revérifier des fichiers déjà installés :

```powershell
.\.venv\Scripts\python.exe scripts/restore_app_artifacts.py --report artifacts/integrity-report.json
```

Conserver une sauvegarde externe des sources : un `git pull` ne restaure pas les
checkpoints ignorés. Les originaux ont été copiés, sans entraînement ni changement
de seuil. CAE Transistor EXP-003 reste une étude complémentaire ; les six modèles
du benchmark officiel utilisent toujours CAE Transistor EXP-001.

## Démarrage

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Choisir Bottle, Hazelnut ou Transistor, puis CAE, PatchCore ou le mode comparaison.
Importer une image PNG/JPEG, lancer l'analyse, consulter les détails et télécharger
le rapport JSON. Les chemins de modèles sont résolus à partir du dépôt ; aucun
chemin Kaggle n'est utilisé à l'exécution. Les modèles sont restaurés sur CPU, sans
téléchargement de backbone, avec `weights_only=True`, chargement strict et contrôle
SHA-256 avant restauration. Une absence de poids produit une erreur explicite.

Un serveur HTTP accessible ne prouve pas que les modèles fonctionnent. La commande
suivante charge réellement les six modèles, compare 26 scores aux résultats figés,
contrôle les banques mémoire, cartes et verdicts, puis teste neuf parcours Streamlit
sans remplacer l'inférence par un modèle factice :

```powershell
.\.venv\Scripts\python.exe scripts/validate_app_release.py --root . `
  --data-root 'CHEMIN_VERS_MVTEC_AD' --output artifacts/release-validation
```

Il suffit de fournir les 13 images `*/test/*` du manifeste
`docs/data/us-05-04/source-images.json`, avec les sous-dossiers d'origine. Les URL
de la version 2 de MVTec AD et les hashes y sont consignés. Le script écrit les
rapports, cartes et superpositions dans le dossier de sortie et échoue si un
contrôle manque. Ne pas l'exécuter avec `python -O`.

## Tests complets et condition de promotion

```powershell
.\.venv\Scripts\python.exe -m pip install -r app/requirements-validation.txt
$env:US0503_ARCHIVE = 'CHEMIN_VERS_us0503-kaggle-370020172.zip'
$env:US0503_ARCHIVE_002 = 'CHEMIN_VERS_results_remainig.zip'
$env:US0504_DATASET = 'CHEMIN_VERS_MVTEC_AD'
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Les deux archives sont identifiées par leur SHA-256 dans
`experiments/benchmark/kaggle-001-provenance.json` et `kaggle-002-provenance.json`.
Le contrôle spatial utilise aussi les masques `ground_truth` du manifeste d'images.
Sans ces fichiers, quatre tests sont ignorés : il faut le signaler explicitement,
pas les présenter comme réussis.

Avant chaque promotion : vérifier les six hashes, les seuils figés, les dépendances,
la suite complète, les inférences réelles, le démarrage HTTP, l'état Git et les règles
GitHub. Comparer de nouveau les têtes distantes et refuser le merge si elles ont
changé depuis les tests. Utiliser une PR/merge traçable, sans push forcé. Ne jamais
inclure les données, environnements, secrets ou modifications locales étrangères.

Les tests d'intégration établissent la fidélité au benchmark, pas une détection
parfaite : les faux négatifs CAE et faux positifs PatchCore documentés subsistent.
