# Artefacts US-05-04 — erreurs, provenance et figures

[Rapport scientifique](../../epic-05/us-05-04.md) · [Galerie légendée](../../epic-05/us-05-04-gallery.md).

| Fichier | Fonction |
|---|---|
| `errors.csv` | Les 86 erreurs image/modèle avec scores, seuils, chemins et SHA NPZ |
| `error-summary.json` | Confusions des six instances et effectifs par type de défaut |
| `selection.json` | Règle de médiane inférieure, 13 cas, rangs, appariement des deux modèles |
| `source-images.json` | 24 PNG externes épinglés : URL versionnée, SHA fichier/pixels, taille/dimensions |
| `spatial-diagnostics.json` | Observations continues sur les 26 cartes, sans seuil pixel |
| `figures/case-01.png` à `case-13.png` | RGB, carte archivée, GT, superposition des deux modèles |
| `figures-manifest.json` | SHA/tailles des sorties, empreintes des entrées et environnement de rendu |

Les données brutes restent **hors dépôt**. Le cache source local utilisé est
`C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0504-work\mvtec-ad-v2`.
Les 13 RGB et 11 GT correspondent à 13 734 107 octets. Les 13 figures font 8 374 373 octets,
chacune inférieure à 713 ko. Les ZIP US-05-03 gardent leurs octets originaux et leurs NPZ.

## Reproduire dans un environnement séparé

Depuis la racine du projet, Python **3.12.6**. Les dépendances de rendu sont figées dans
[requirements-us0504.txt](../../../scripts/requirements-us0504.txt) : NumPy 2.5.5,
Pillow 13.0.0, Matplotlib 3.11.2 et ses dépendances. Les contrôles d'inventaire et de provenance
fonctionnent aussi dans le `.venv` scientifique existant avec NumPy 2.5.4 ; les versions figées
servent à la reproduction PNG exacte. Le backend est Agg et la police DejaVu Sans de Matplotlib.

```powershell
py -3.12 -m venv 'C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0504-work\.venv'
$analysisPython = 'C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0504-work\.venv\Scripts\python.exe'
& $analysisPython -m pip install -r scripts/requirements-us0504.txt

$env:US0503_ARCHIVE = 'C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0503-kaggle-370020172.zip'
$env:US0503_ARCHIVE_002 = 'C:\Users\marwa\Downloads\results_remainig.zip'
$env:US0504_DATASET = 'C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0504-work\mvtec-ad-v2'

# Vérification légère sans réseau, RGB ou NPZ externes :
& $analysisPython -B scripts/analyze_benchmark_errors.py check

# Récupération des 24 PNG manquants seulement ; octets comparés aux SHA épinglés :
& $analysisPython -B scripts/analyze_benchmark_errors.py fetch

# Reproduction dans un dossier temporaire et comparaison binaire des sorties :
& $analysisPython -B scripts/analyze_benchmark_errors.py reproduce
```

Pour une régénération explicite des dérivés uniquement, `inventory` réécrit les trois inventaires
et `render` réécrit les figures/diagnostics et leur manifeste. Ces commandes ne modifient jamais
les benchmarks ou les seuils. `check` et `reproduce` ne modifient aucun artefact versionné.
Un changement des données sources, des références ou des masques provoque un échec.

La reproduction testée a utilisé les bibliothèques figées dans un dossier isolé via `--target`
et `PYTHONPATH`, sans modifier le `.venv` scientifique. Les 13 PNG et les diagnostics ont été
reproduits **octet pour octet** sur Windows. D'autres versions de bibliothèques/polices/OS peuvent
modifier les octets des figures ; elles doivent être déclarées, sans contourner les contrôles.
Ne versionner ni l'environnement séparé ni le cache de données.

## Tests

Dans le `.venv` existant (pytest 9.2.0), avec les trois variables externes définies :

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/evaluation/test_error_analysis.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m pytest tests/evaluation tests/data/test_data_pipeline.py -q -p no:cacheprovider
```

Résultats du 16 novembre 2026 : **27 tests nouveaux réussis**, puis **323 tests et 2 sous-tests**,
aucun ignoré avec les sources externes disponibles. Sans `US0504_DATASET` ou l'un des ZIP,
le test externe US-05-04 est explicitement ignoré ; aucun téléchargement n'est effectué par pytest.
Les tests d'archives US-05-03 indiquent aussi leurs propres prérequis manquants. Les tests des
petits fichiers seuls ne constituent pas un nouvel audit de disponibilité des sources externes.

## Licence et limite de provenance

MVTec AD : Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger, MVTec Software GmbH,
CVPR 2019. [Source officielle](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
[redistribution Kaggle ipythonx v2](https://www.kaggle.com/datasets/ipythonx/mvtec-ad/versions/2).
Données et figures dérivées : **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.
Transformations déclarées : redimensionnement, palette, superposition et légendes. Les images
ne sont pas retouchées pour fabriquer un défaut. La licence des images ne relicencie pas le code.

US-05-03 ne consignait pas les SHA des RGB historiques. Les nouveaux SHA épinglent les fichiers
récupérés ; même chemin/version/catégorie et égalité des GT avec les deux NPZ sont vérifiés,
mais cela ne crée pas rétrospectivement une preuve binaire indépendante des anciens RGB d'entrée.
