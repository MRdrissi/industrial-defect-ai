# Benchmarks Test — US-05-03

**Livraison technique complète 6/6**, sans relancer les modèles ni modifier les seuils.
[Rapport comparatif, audit et DoD](../../docs/epic-05/us-05-03.md) ·
[Table des six instances](comparison.csv) · [Synthèse et macros](summary.json).

| Lot | Instances | Version Kaggle | Prédictions / NPZ audités | Petits originaux |
|---|---|---:|---:|---:|
| [kaggle-001](kaggle-001/) | CAE Transistor EXP-001, PatchCore Bottle EXP-002 | 370020172 | 183 / 183 | 6 |
| [kaggle-002](kaggle-002/) | CAE Bottle EXP-002, CAE Hazelnut EXP-001, PatchCore Hazelnut EXP-002 corrigée, PatchCore Transistor EXP-001 | 370053025 | 403 / 403 | 12 |

Les **18 CSV/JSON originaux** sont conservés octet pour octet, protégés par `.gitattributes -text`.
Le premier lot et sa [provenance historique](kaggle-001-provenance.json) restent inchangés depuis `5038988` ;
son statut `partial_2_of_6` décrit cette livraison historique. La [provenance du deuxième lot](kaggle-002-provenance.json)
référence les douze fichiers, les quatre manifestes et le ZIP original. Les six rapports déclarent
le commit scientifique `0acbf26880b755373f2e61236d009ed16f04658c` et les seuils US-05-02 figés.

Les NPZ restent dans les Outputs Kaggle versionnés et les archives locales hors Git :

| Archive originale | Octets | SHA-256 | Source |
|---|---:|---|---|
| `us0503-kaggle-370020172.zip` | 93019917 | `3f847e7bcbda2816ebb5ef4f41cd52a6720bd758b877a4a8f6f753d9ff7fcd45` | [Outputs 370020172](https://www.kaggle.com/code/marwanedrissi/industrialdefect-ai-test-benchmark-001/output?scriptVersionId=370020172) |
| `results_remainig.zip` | 153887769 | `85d9ff82c22126c181431dda845efeea56578ca66809d53e60b5ed61da4fe879` | [Outputs 370053025](https://www.kaggle.com/code/marwanedrissi/industrialdefect-ai-test-benchmark-002-v1/output?scriptVersionId=370053025) |

Les deux archives sont conservées sans modification. Les contrôles démontrent la présence des
fichiers sauvegardés dans les téléchargements ; la correspondance de la deuxième archive à la version
est fournie par l'utilisateur et recoupée avec les quatre ancres. Pas de nouvelle inspection distante
authentifiée lors de cette intégration. L'ancien affichage « empty » du premier lot est résolu par
les fichiers réellement présents dans son ZIP. L'accès futur dépend de Kaggle ou de la conservation
locale des archives. Aucun NPZ de benchmark, ZIP, dataset, checkpoint, cache ou clone Kaggle n'est ajouté à Git.

Les audits vérifient SHA/CRC, structure sûre, couverture Test, indices, liens CSV/manifeste/NPZ,
seuils hex, métadonnées scientifiques, masques binaires et cartes finies. Les **293 masques** sont
identiques entre familles. Image-AUROC, décisions et latences sont recoupées avec les CSV ;
les Pixel-AUROC restent les valeurs originales sous empreinte, sans recalcul.

## Synthèses dérivées

`comparison.csv` contient les six résultats à pleine précision et leurs sources.
`summary.json` contient les macro-moyennes, **(Bottle + Hazelnut + Transistor) / 3**, et les décomptes
des métriques définies/indéfinies (3/0 partout). Le F1 est moyenné directement ; la latence macro
est la moyenne des trois latences moyennes. Aucune micro-moyenne, aucun score inter-instance regroupé,
aucun percentile global reconstitué. Mesures à une exécution par instance sur Tesla T4, sans
intervalle de confiance entre exécutions ; périmètre d'inférence uniquement.

```powershell
.\.venv\Scripts\python.exe -B scripts/summarize_benchmark.py --check
# Régénération explicite des deux seuls dérivés, si nécessaire :
.\.venv\Scripts\python.exe -B scripts/summarize_benchmark.py --write
```

## Répéter les contrôles

```powershell
$env:US0503_ARCHIVE = 'C:\Users\marwa\Documents\ChatGPT\Projet DeepLearning\us0503-kaggle-370020172.zip'
$env:US0503_ARCHIVE_002 = 'C:\Users\marwa\Downloads\results_remainig.zip'
.\.venv\Scripts\python.exe -B -m pytest tests/evaluation/test_benchmark_artifacts.py tests/evaluation/test_benchmark_summary.py -q -p no:cacheprovider
```

Résultat : **39 tests réussis**, aucun ignoré avec les deux archives. Sans archives, trois tests
spatiaux sont explicitement ignorés ; les contrôles des petits fichiers seuls ne prouvent pas
une nouvelle disponibilité des NPZ. Le rapport détaille aussi la validation élargie :
**296 tests et 2 sous-tests réussis**. Aucun calcul de modèle ou ajustement de seuil n'est exécuté.
