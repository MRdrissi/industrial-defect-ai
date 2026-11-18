# EPIC-02 — Dataset & Experimental Protocol

**Projet :** IndustrialDefect AI  
**Branche :** `epic/02-data-preparation`  
**Jira :** Epic **IDL-2** — projet *IndustrialDefect Deep Learning* (IDL)  
**État :** US-02-01 à US-02-04 documentées.
**Public :** tous les membres de l'équipe et toute personne devant reproduire la préparation des données.

> **Objectif de cet Epic :** valider MVTec AD sur trois catégories obligatoires (`bottle`, `transistor`, `hazelnut`), fixer les splits train/validation/test sans fuite d'information, et préparer images et masques avec un pipeline reproductible commun aux deux approches (Convolutional Autoencoder et PatchCore).

---

## 1. User Stories de l'Epic

| US | Jira | Titre | Responsable | Dépend de | Section |
|---|---|---|---|---|---|
| US-02-01 | IDL-9 | Acquérir, vérifier et documenter les trois catégories MVTec AD | Zakaria El Bergui (M1) | — | [§ 2](#2-us-02-01--acquérir-vérifier-et-documenter-les-trois-catégories-mvtec-ad) |
| US-02-02 | IDL-12 | Explorer les images, anomalies et masques des trois catégories | Zakaria El Bergui (M1) | US-02-01 | [§ 3](#3-us-02-02--explorer-les-images-anomalies-et-masques) |
| US-02-03 | IDL-15 | Définir et figer le protocole Train/Validation/Test | Mouad Fekkak (M2) | US-02-01 | [Protocole](epic-02/us-02-03.md) |
| US-02-04 | IDL-18 | Développer et tester le pipeline commun de prétraitement | Marwane Drissi (M3) | US-02-02, US-02-03 | [Pipeline](epic-02/us-02-04.md) |

**Règle commune :** chaque US ajoute sa propre section dans ce document. Les données de **test** ne doivent jamais servir à ajuster un modèle, un seuil ou un hyperparamètre.

---

## 2. US-02-01 — Acquérir, vérifier et documenter les trois catégories MVTec AD

**Jira :** IDL-9 · sous-tâches **IDL-10** (ST-02-01-01, vérification) et **IDL-11** (ST-02-01-02, documentation)

### Critères d'acceptation et preuves

| # | Critère | Preuve | État |
|---|---|---|---|
| 1 | Les trois catégories et leurs répertoires `train/`, `test/`, `ground_truth/` sont accessibles | Notebook, étape 3 : 15 contrôles sur 15 réussis | ✅ |
| 2 | Les effectifs normaux/anormaux et les masques correspondants sont comptés et vérifiés | Notebook, étapes 4 et 5 : effectifs conformes aux chiffres officiels, 0 problème d'intégrité | ✅ |
| 3 | Source, version/licence, chemins configurables et procédure de reproduction documentés | Présente section (§ 2.1 à § 2.8) | ✅ |

### 2.1. Source et provenance des données

| Élément | Valeur |
|---|---|
| Jeu de données | **MVTec Anomaly Detection (MVTec AD)** |
| Producteur original | MVTec Software GmbH |
| Page officielle | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| Référence scientifique | P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, IEEE CVPR 2019 |
| Licence | **CC BY-NC-SA 4.0** — usage non commercial uniquement ; attribution et partage dans les mêmes conditions |
| **Copie utilisée (Kaggle)** | https://www.kaggle.com/datasets/ipythonx/mvtec-ad |
| Auteur de la copie Kaggle | Innat (`ipythonx`) |
| **Version Kaggle** | **Version 2** — 5,27 Go, 15 catégories, mise à jour il y a environ 5 ans |
| Fichiers officiels inclus | `readme.txt` (citation) et `license.txt` (licence) à la racine |

**Pourquoi cette copie ?** Elle contient les **15 catégories complètes et non modifiées** (pas de redimensionnement ni d'augmentation), inclut les fichiers officiels de MVTec et est largement utilisée (environ 47 000 téléchargements, plus de 360 notebooks). Ses effectifs ont été vérifiés un par un contre l'article de référence (§ 2.4).

**Licence — points d'attention :** les images ne doivent **pas** être ajoutées au dépôt GitHub ni redistribuées. Le dossier `/data/` est d'ailleurs ignoré par Git (`.gitignore`). Toute publication (rapport, démonstration) doit citer l'article de référence.

### 2.2. Procédure de reproduction (accès au dataset)

Pour retrouver exactement les mêmes données sur son propre compte Kaggle :

1. Se connecter à Kaggle (compte personnel avec téléphone vérifié).
2. **Create → New Notebook**, puis **Session options → Accelerator → None** : la vérification n'a pas besoin de GPU.
3. Panneau **Input → Add Input → Datasets**, rechercher **`mvtec ad`** et ajouter le dataset **MVTec AD** de **Innat** (`ipythonx/mvtec-ad`, environ 5 Go).
4. Importer le notebook du dépôt : **File → Import Notebook** → `notebooks/02-mvtec-dataset-check.ipynb`.
5. **Run All**. Chaque contrôle se termine par un `assert` : le notebook s'arrête si une vérification échoue.
6. Pour conserver les résultats : **Save Version → Save & Run All (Commit)**. Les fichiers sont alors disponibles dans l'onglet **Output** (`mvtec_check/`).

**Résultat attendu :** `STRUCTURE VALIDE … True`, `EFFECTIFS CONFORMES AUX CHIFFRES OFFICIELS : True`, `Images ouvertes avec succès : 1106`, `INTÉGRITÉ VALIDÉE : True`.

### 2.3. Chemins configurables

Aucun chemin n'est écrit « en dur ». La première cellule de code du notebook regroupe la configuration :

```python
KAGGLE_INPUT = Path("/kaggle/input")
DATASET_ROOT = None          # None = détection automatique ; sinon, mettre un chemin
CATEGORIES = ["bottle", "hazelnut", "transistor"]
REQUIRED_DIRS = ["train", "test", "ground_truth"]
```

| Paramètre | Rôle | Valeur observée |
|---|---|---|
| `DATASET_ROOT` | Racine du dataset. Avec `None`, la fonction `find_mvtec_root()` cherche sous `KAGGLE_INPUT` le dossier contenant `bottle/train`, `bottle/test` et `bottle/ground_truth`, et s'arrête si elle en trouve zéro ou plusieurs | `/kaggle/input/datasets/ipythonx/mvtec-ad` |
| `CATEGORIES` | Catégories contrôlées. Ajouter une catégorie (par ex. une extension) = modifier cette liste et ajouter ses effectifs dans `EXPECTED` | `bottle`, `hazelnut`, `transistor` |
| `KAGGLE_INPUT` | Dossier de recherche. À remplacer par un dossier local pour une exécution hors Kaggle | `/kaggle/input` |
| `OUTPUT_DIR` (étape 6) | Dossier des résultats | `/kaggle/working/mvtec_check` |

**Pourquoi la détection automatique ?** Le chemin de montage des datasets Kaggle a déjà changé par le passé et peut différer d'un compte à l'autre. La détection rend le notebook exécutable chez chaque membre de l'équipe sans modification.

**Exécution locale (optionnelle) :** télécharger le dataset depuis la page officielle ou Kaggle, l'extraire dans `<racine-du-dépôt>/data/mvtec_ad/` (dossier ignoré par Git), puis définir `KAGGLE_INPUT = Path("../data")`.

### 2.4. Structure et effectifs vérifiés

Structure commune aux trois catégories :

```text
<catégorie>/
├── train/good/                 # images normales uniquement (apprentissage)
├── test/good/                  # images normales (évaluation)
├── test/<type_de_défaut>/      # images défectueuses (évaluation)
└── ground_truth/<type_de_défaut>/<id>_mask.png   # un masque par image défectueuse
```

**Effectifs détaillés** — identiques aux chiffres officiels de Bergmann et al. (CVPR 2019) :

| Catégorie | `train/good` | `test/good` | Défauts du test (images = masques) | Total défauts |
|---|---:|---:|---|---:|
| bottle | 209 | 20 | broken_large 20 · broken_small 22 · contamination 21 | 63 |
| hazelnut | 391 | 40 | crack 18 · cut 17 · hole 18 · print 17 | 70 |
| transistor | 213 | 60 | bent_lead 10 · cut_lead 10 · damaged_case 10 · misplaced 10 | 40 |

**Résumé :**

| Catégorie | Normales (train + test) | Anormales | Masques |
|---|---:|---:|---:|
| bottle | 229 | 63 | 63 |
| hazelnut | 431 | 70 | 70 |
| transistor | 273 | 40 | 40 |
| **Total** | **933** | **173** | **173** |

Soit **1 106 images** et **173 masques** pour les trois catégories.

### 2.5. Intégrité et formats

Contrôles réalisés sur **chaque** fichier :

- chaque image s'ouvre et se **décode entièrement** (`PIL.Image.load()`), ce qui détecte les fichiers corrompus ou tronqués ;
- chaque image défectueuse possède **son** masque (`000.png` → `000_mask.png`) ;
- chaque masque a **la même taille** que son image, est **binaire** (0 = normal, 255 = défaut) et **non vide**.

**Résultat :** 1 106 images lisibles, **0 problème détecté**.

| Catégorie | Taille | Mode | Images |
|---|---|---|---:|
| bottle | 900 × 900 | RGB | 292 |
| hazelnut | 1024 × 1024 | RGB | 501 |
| transistor | 1024 × 1024 | RGB | 313 |

**Conséquence pour le prétraitement (US-02-04) :** les tailles diffèrent entre catégories. Un redimensionnement commun sera nécessaire, appliqué de la même façon aux images et aux masques (interpolation « plus proche voisin » pour les masques, afin qu'ils restent binaires).

### 2.6. Environnement d'exécution

| Élément | Valeur |
|---|---|
| Plateforme | Kaggle Notebook, sans accélérateur (CPU) |
| Python | 3.13.18 |
| pandas | 2.3.3 |
| numpy | 2.1.3 |
| Pillow | 13.0.0 |
| Système | Linux-6.18.55+-x86_64-with-glibc2.39 |
| Date d'exécution | 2026-11-09 19:48:34 UTC |

Le poste local de l'équipe utilise Python 3.12 (voir EPIC-01). La vérification n'utilise que pandas, numpy et Pillow : elle ne dépend pas de PyTorch.

### 2.7. Artefacts et références

| Artefact | Emplacement |
|---|---|
| Notebook de vérification | [`notebooks/02-mvtec-dataset-check.ipynb`](../notebooks/02-mvtec-dataset-check.ipynb) — commit `6fcda72` |
| Version Kaggle exécutée | Version #1 — https://www.kaggle.com/code/zakariaelbergui/02-mvtec-dataset-check?scriptVersionId=367806648 |
| Effectifs détaillés | [`docs/data/us-02-01/counts_detailed.csv`](data/us-02-01/counts_detailed.csv) |
| Résumé des effectifs | [`docs/data/us-02-01/counts_summary.csv`](data/us-02-01/counts_summary.csv) |
| Formats des images | [`docs/data/us-02-01/image_formats.csv`](data/us-02-01/image_formats.csv) |
| Rapport complet (source, licence, contrôles, environnement) | [`docs/data/us-02-01/dataset_check_report.json`](data/us-02-01/dataset_check_report.json) |

Les fichiers de `docs/data/us-02-01/` sont les sorties exactes de la Version #1 Kaggle (dossier `mvtec_check/` de l'onglet Output).

### 2.8. Limites et points d'attention

- La copie Kaggle n'est pas distribuée par MVTec elle-même. Sa conformité a été établie par comparaison des effectifs avec l'article de référence et par la présence des fichiers officiels `readme.txt` et `license.txt`.
- Cette US vérifie la **présence** et l'**intégrité** des données. Elle n'analyse pas leur contenu visuel (rôle de US-02-02) et ne définit pas les splits (rôle de US-02-03).
- `test/` contient à la fois des images normales et défectueuses. Le découpage validation/test et l'interdiction d'utiliser le test pour régler les modèles relèvent du protocole de US-02-03.

---

## 3. US-02-02 — Explorer les images, anomalies et masques

**Jira :** IDL-12 · sous-tâches **IDL-13** (ST-02-02-01, visualisation) et **IDL-14** (ST-02-02-02, documentation)

### Critères d'acceptation et preuves

| # | Critère | Preuve | État |
|---|---|---|---|
| 1 | Un notebook EDA exécuté présente des images normales, anormales et des superpositions de masques pour les trois catégories | `notebooks/03-mvtec-eda.ipynb`, étapes 3 et 4 ; figures `01_*` et `02_*` | ✅ |
| 2 | Distribution des dimensions, sous-types de défauts et principales difficultés documentés | § 3.2 à § 3.5 ; statistiques sur les 173 masques | ✅ |
| 3 | Observations utiles au prétraitement et à la localisation consignées | § 3.6 | ✅ |

**Données utilisées :** même copie que US-02-01 (`ipythonx/mvtec-ad`, Version 2). Les exemples affichés sont tirés au hasard avec une **seed fixe** (42 pour les images normales, 43 pour les défauts) : tout le monde obtient les mêmes images.

**Règle respectée :** les images de **test** (défauts et masques) servent uniquement à **décrire** les anomalies. Aucun modèle, seuil ni hyperparamètre n'est ajusté dans cette US.

### 3.1. Images normales : ce que les modèles apprendront

![Images normales](data/us-02-02/figures/01_normal_examples.png)

| Catégorie | Position / orientation | Fond | Variabilité entre pièces normales |
|---|---|---|---|
| bottle | Vue de dessus, toujours centrée | Clair, uniforme | Très faible |
| hazelnut | **Orientation et position aléatoires** | Noir, uniforme | Forte (forme, teinte, position de la base claire) |
| transistor | **Toujours dans le même sens**, pattes vers le bas | Plaque perforée (texture répétitive) | Faible (courbure des pattes, léger décalage) |

### 3.2. Dimensions des images

Toutes les images d'une même catégorie ont la même taille, et chaque masque a la taille de son image (vérifié en US-02-01).

| Catégorie | Taille | Mode | Images |
|---|---|---|---:|
| bottle | 900 × 900 | RGB | 292 |
| hazelnut | 1024 × 1024 | RGB | 501 |
| transistor | 1024 × 1024 | RGB | 313 |

### 3.3. Types de défauts : exemples

Un exemple par type de défaut : image, masque (blanc = défaut) et superposition en rouge.

| bottle | hazelnut | transistor |
|---|---|---|
| ![bottle](data/us-02-02/figures/02_defects_bottle.png) | ![hazelnut](data/us-02-02/figures/02_defects_hazelnut.png) | ![transistor](data/us-02-02/figures/02_defects_transistor.png) |

| Catégorie | Défaut | Nature observée |
|---|---|---|
| bottle | broken_large | Grande cassure du goulot, bien visible |
| bottle | broken_small | Petit éclat sur le bord |
| bottle | contamination | Trace à l'intérieur, **peu visible** à l'œil nu |
| hazelnut | crack | Zone fissurée de la coque |
| hazelnut | cut | Coupure **fine et linéaire** |
| hazelnut | hole | Petit trou rond, bien contrasté |
| hazelnut | print | Inscription imprimée sur la coque, en **plusieurs fragments** |
| transistor | bent_lead | Patte tordue : seul le bout des pattes est marqué |
| transistor | cut_lead | Patte coupée |
| transistor | damaged_case | Boîtier abîmé |
| transistor | misplaced | Composant mal placé ou absent : le masque couvre **toute la zone attendue** du composant (défaut **logique**, pas de texture) |

### 3.4. Taille des défauts (173 masques)

Mesures réalisées sur **toutes** les images défectueuses : surface (part de l'image), nombre de régions séparées (`scipy.ndimage.label`) et boîte englobante.

| Catégorie | Défaut | Images | Surface min | Surface médiane | Surface max | Régions max | Plus petit côté de la boîte à 256 px (min) |
|---|---|---:|---:|---:|---:|---:|---:|
| bottle | broken_large | 20 | 4,09 % | 11,87 % | 27,42 % | 2 | 59,2 px |
| bottle | broken_small | 22 | 0,88 % | 2,44 % | 8,06 % | 3 | 27,3 px |
| bottle | contamination | 21 | 0,58 % | 7,25 % | 16,53 % | 2 | 10,5 px |
| hazelnut | crack | 18 | 1,28 % | 4,70 % | 27,66 % | 4 | 40,0 px |
| hazelnut | cut | 17 | 0,40 % | 0,79 % | 2,02 % | 4 | 10,5 px |
| hazelnut | hole | 18 | 0,23 % | 1,24 % | 2,93 % | 3 | 13,5 px |
| hazelnut | print | 17 | 1,52 % | 2,44 % | 4,83 % | 15 | 31,8 px |
| transistor | bent_lead | 10 | 0,22 % | 1,03 % | 3,65 % | 2 | 11,2 px |
| transistor | cut_lead | 10 | 1,02 % | 1,48 % | 3,58 % | 2 | 23,2 px |
| transistor | damaged_case | 10 | 0,63 % | 2,31 % | 5,35 % | 2 | 22,0 px |
| transistor | misplaced | 10 | 37,73 % | 40,45 % | 49,32 % | 1 | 138,8 px |

| Catégorie | Défauts | Surface moyenne | Surface médiane | Min | Max |
|---|---:|---:|---:|---:|---:|
| bottle | 63 | 7,62 % | 5,72 % | 0,58 % | 27,42 % |
| hazelnut | 70 | 3,35 % | 1,62 % | 0,23 % | 27,66 % |
| transistor | 40 | 11,98 % | 2,03 % | 0,22 % | 49,32 % |

![Distribution de la taille des défauts](data/us-02-02/figures/03_defect_area_distribution.png)

**Chiffres clés :**
- **27 défauts sur 173 (16 %)** couvrent **moins de 1 %** de l'image.
- La surface varie de **0,22 %** (`bent_lead`) à **49,32 %** (`misplaced`), soit un rapport supérieur à 200.
- À **256 × 256**, **aucun** défaut ne passe sous 4 px : le plus petit côté de boîte englobante reste d'au moins **10,5 px**.

### 3.5. Principales difficultés

1. **Petits défauts** : `hazelnut cut` (médiane 0,79 %), `hole`, `transistor bent_lead`. La **localisation** au pixel près sera difficile, et la **Pixel-AUROC** sera plus exigeante que l'Image-AUROC.
2. **Défauts peu contrastés** : `bottle contamination` est à peine visible ; le modèle doit détecter des variations fines de texture.
3. **Défauts fragmentés** : `print` (jusqu'à 15 régions), `crack` et `cut` (jusqu'à 4). La carte d'anomalies doit pouvoir signaler **plusieurs zones** dans une même image.
4. **Défaut logique** : `misplaced` ne se voit pas comme une texture anormale mais comme une **position** anormale. Les méthodes à base de patches locaux (PatchCore) peuvent y être moins sensibles qu'aux défauts de texture : à surveiller en EPIC-03.
5. **Variabilité normale de hazelnut** : l'orientation et l'aspect des noisettes normales varient beaucoup ; le modèle doit les accepter comme normaux sans masquer les vrais défauts.
6. **Grande dispersion des tailles** dans un même type (`contamination`, `crack`, `bent_lead`) : un seuil unique devra fonctionner pour des défauts minuscules comme pour des défauts étendus.

### 3.6. Observations pour le prétraitement et la localisation (entrée de US-02-04)

| Sujet | Observation | Recommandation |
|---|---|---|
| Taille d'entrée | Tailles différentes (900 et 1024 px) ; à 256 px, le plus petit défaut garde ≥ 10,5 px de côté de boîte ; à 128 px, il tomberait vers 5 px | Redimensionner **toutes** les catégories à une taille commune ; **256 × 256** paraît suffisant, éviter 128 × 128 |
| Masques | Binaires (0/255), même taille que l'image | Redimensionner avec une interpolation **plus proche voisin** puis re-binariser, pour garder des masques 0/1 exacts |
| Rotation / retournement | bottle et hazelnut sont invariants à la rotation ; **transistor ne l'est pas** (une rotation ou un décalage imite `misplaced`) | Augmentations géométriques autorisées pour bottle et hazelnut uniquement ; **aucune** pour transistor |
| Couleur / luminosité | Fonds et éclairages homogènes dans chaque catégorie ; `contamination` repose sur de faibles variations | Augmentations de couleur **légères** au plus, pour ne pas effacer les défauts peu contrastés |
| Recadrage | Les défauts peuvent se situer au bord (goulot de bottle, pattes de transistor) | Éviter les recadrages aléatoires qui couperaient des zones utiles |
| Métriques | Défauts petits, fragmentés, et un défaut logique étendu | Évaluer séparément **Image-AUROC** et **Pixel-AUROC**, et analyser les résultats **par type de défaut** |

### 3.7. Environnement et artefacts

| Élément | Valeur |
|---|---|
| Plateforme | Kaggle Notebook, sans accélérateur (CPU) |
| Python / matplotlib / SciPy | 3.13.18 / 3.10.0 / version de l'image Kaggle |
| Seeds | 42 (images normales), 43 (exemples de défauts) |
| Notebook | [`notebooks/03-mvtec-eda.ipynb`](../notebooks/03-mvtec-eda.ipynb) |
| Version Kaggle (visualisation) | Version #1 — https://www.kaggle.com/code/zakariaelbergui/03-mvtec-eda?scriptVersionId=367819083 |
| Version Kaggle (statistiques) | Version #2 — https://www.kaggle.com/code/zakariaelbergui/03-mvtec-eda?scriptVersionId=367823089 |
| Figures | [`docs/data/us-02-02/figures/`](data/us-02-02/figures/) — 5 PNG |
| Mesures par image | [`docs/data/us-02-02/defect_stats_per_image.csv`](data/us-02-02/defect_stats_per_image.csv) |
| Résumé par type de défaut | [`docs/data/us-02-02/defect_summary_per_type.csv`](data/us-02-02/defect_summary_per_type.csv) |

**Limite :** le « plus petit côté de la boîte englobante » mesure le rectangle autour du défaut. Pour un défaut fin et oblique (`cut`), le trait lui-même est plus fin que la boîte : ces défauts restent les plus fragiles à la réduction de résolution.

## 4. US-02-03 — Protocole Train/Validation/Test

Les trois manifestes figent 650 images normales pour l’entraînement, 163 pour la validation et 293 images du test officiel. [Protocole, empreintes et contrôles de séparation](epic-02/us-02-03.md)

## 5. US-02-04 — Pipeline commun de prétraitement

Le pipeline commun lit les manifestes et fournit les images RGB, labels et masques aux profils CAE et PatchCore. Sur Kaggle, le contrôle du profil CAE à 256 × 256 couvre 1 106 images et 173 masques d’anomalies. Aucun masque n’a entièrement disparu. [API, réutilisation et preuves de validation](epic-02/us-02-04.md)

---

## Références

- [MVTec AD — page officielle et conditions d'utilisation](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [MVTec AD — copie Kaggle utilisée (`ipythonx/mvtec-ad`, Version 2)](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
- P. Bergmann *et al.*, *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
- [Licence CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
