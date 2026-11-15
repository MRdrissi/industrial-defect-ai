# Benchmarks Test — US-05-03

**Intégration partielle 2/6 ; Story ouverte.** Seuls CAE Transistor EXP-001 et PatchCore Bottle
EXP-002 sont archivés. [Résultats, protocole et validation](../../docs/epic-05/us-05-03.md).

Les six originaux de [kaggle-001/](kaggle-001/) sont extraits sans réécriture de la version
Kaggle **370020172**, commit scientifique `0acbf26880b755373f2e61236d009ed16f04658c`.
La [provenance](kaggle-001-provenance.json) est séparée des fichiers scientifiques.

| Fichier original | Octets | SHA-256 |
|---|---:|---|
| `cae-transistor/manifest.json` | 19016 | `b46e66ff9f11f6af17ae0d453490217bca537e7a1f020348d2fa3a0060f89e19` |
| `cae-transistor/metrics.json` | 2983 | `c6b8a9ff01f98783ce70857e01d9420660a0da31a0dc7b4ae77f61ba6f9b44b7` |
| `cae-transistor/predictions.csv` | 16205 | `7323d82cc76eec12b3e4b49361b3136555300b09fadc1ad418252d6a8522bdbd` |
| `patchcore-bottle/manifest.json` | 15845 | `009e3c2153d6c5f8a98ede447e0bb215d4cbf4820ff6ac1e377f63df59dee56e` |
| `patchcore-bottle/metrics.json` | 3091 | `2b6f198c1de9420d76a9ed58fcf9fff81ba492523bedd1599037f8ac242ccab6` |
| `patchcore-bottle/predictions.csv` | 13256 | `e994eb753c7e9fe0d81cfe2b3f52e3f1b4b0a2b0286199ff5d8e426264de9c10` |

Les manifestes et CSV conservent les 183 chemins et empreintes NPZ. Vérification complète du
snapshot téléchargé : **100 NPZ CAE (26 985 219 octets) + 83 NPZ PatchCore (21 224 541 octets)**,
sans fichiers manquants. L’affichage historique « empty » est levé par le contrôle des octets.
Les tableaux `mask` et `anomaly_map` sont vérifiés sans recalculer les métriques.

Les NPZ restent hébergés dans les [Outputs Kaggle versionnés](https://www.kaggle.com/code/marwanedrissi/industrialdefect-ai-test-benchmark-001/output?scriptVersionId=370020172)
et dans `us0503-kaggle-370020172.zip`, téléchargé hors dépôt (93 019 917 octets ; SHA-256
`3f847e7bcbda2816ebb5ef4f41cd52a6720bd758b877a4a8f6f753d9ff7fcd45`).
La totalité du ZIP n'est pas une livraison Git : son clone Kaggle et les autres fichiers
ne sont pas intégrés. Aucun dataset, checkpoint, cache, ZIP ou NPZ n'est versionné ici.

Pour répéter le contrôle spatial, fournir le ZIP via `US0503_ARCHIVE` puis lancer uniquement
`tests/evaluation/test_benchmark_artifacts.py`. Sans archive, le test externe est ignoré
explicitement ; les sept tests des petits artefacts restent disponibles. Aucun calcul
de métriques, aucune inférence ni recalibration. Aucune préparation des quatre autres instances.
