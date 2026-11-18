# Validation DefectVision — 18 novembre 2026

**Conclusion : contrôles de publication réussis.** Les six modèles réels sont
restaurés, vérifiés et exécutés. Ce rapport constitue la preuve préalable à la
promotion, dont les PR et commits de merge sont enregistrés dans GitHub.

## État initial et périmètre

- Dépôt local : `C:/Projects/industrial-defect-ai`, branche `develop` à `0e102a0`.
- GitHub `develop` après fetch : `ecc3d9369ff1ab316d67e5144048862fbe750b2c` (cinq commits supplémentaires).
- `main` initial : `059e15616e7917ddf6c58f63caa5e5704d85495c`.
- Notebook local modifié : `notebooks/03-mvtec-eda.ipynb`, conservé hors des commits.
  SHA-256 avant/après : `4af5f92be60ccf2065dbed0f338c35ac455910833cb47f7eb09fa2016414618f`.
- Validation dans une copie isolée du dernier `develop`. Aucun changement de poids,
  d'architecture, de prétraitement, de résultats scientifiques ou de seuils.

## Restauration et corrections

Les six chemins `artifacts/` étaient absents. Les fichiers exacts ont été retrouvés
dans les exports locaux et copiés dans le dépôt principal et la copie de validation.
Total : 350,282,506 octets. Aucune source écrasée ni effacée.
Les sources et membres ZIP sont détaillés dans [restoration.json](restoration.json).

Anomalib manquait. Installation de la version 2.7.0 et de ses dépendances avec
contraintes conservant toutes les versions préexistantes. `pip check` réussi ;
le fichier d'installation correspond à l'environnement testé. Le verrouillage
contient 95 distributions pour Windows/Python 3.12 CPU. Source de référence :
[Anomalib 2.7.0 sur PyPI](https://pypi.org/project/anomalib/2.7.0/).

Livrables ajoutés : dépendances complètes, restauration par hash sans écrasement,
validation réelle reproductible, tests des refus de restauration et
[guide de lancement](../../RUNNING_DEFECTVISION.md).

## Tests et portée des preuves

| Contrôle | Résultat |
|---|---|
| Suite complète avec les deux archives NPZ et les RGB/GT | **529 réussis, 0 ignoré, 23 sous-tests réussis** |
| Checkpoints CAE / PatchCore | **6/6 SHA-256 conformes** |
| Banques mémoire PatchCore | **3/3 restaurées, dimensions conformes, valeurs finies** |
| Inférences sur images MVTec AD v2 originales vérifiées | **26/26**, scores, cartes 256×256 et verdicts conformes |
| Interface Streamlit sans simulation d'inférence | **9/9** : six inspections et trois comparaisons ; rapports, détails, retour et état du bouton contrôlés |
| Serveur réel HTTP local | `/_stcore/health` = 200 / `ok`, page `/` = 200 ; processus de test arrêté |
| Dépendances | `pip check` et résolution du fichier de dépendances réussis |
| Préservation du notebook local | SHA-256 inchangé |

Les 26 inférences couvrent les 13 images de test déjà sélectionnées pour l'analyse
d'erreurs US-05-04, avec les deux familles. Comparaison aux scores Kaggle archivés :
tolérance relative `1e-4`, absolue `1e-5` ; écart absolu maximal observé : **5.722045898e-05**.
Ce contrôle ne constitue pas une nouvelle évaluation intégrale des 293 images.
Les tests d'archives vérifient les 586 couples carte/masque sauvegardés et la
concordance des 293 masques entre familles. Aucun seuil n'a été réestimé.

Les faux négatifs du CAE et faux positifs de PatchCore connus sur ces cas sont
reproduits, pas corrigés artificiellement. La fidélité d'exécution ne signifie
pas que chaque défaut est détecté. EXP-003 Transistor reste complémentaire et
n'est pas substituée à EXP-001 dans le benchmark officiel.

Preuves machine : [pytest.xml](pytest.xml), [sortie pytest](pytest.txt),
[inférences et interface](real-validation.json), [serveur HTTP](http-startup.json).
Les SHA-256 des fichiers Python effectivement testés figurent dans le rapport
d'inférence ; ils permettent de vérifier la correspondance du code publié.

## Checkpoints et seuils

| Instance | SHA-256 du fichier effectivement restauré | Seuil |
|---|---|---:|
| cae-bottle | `12501c4d125e8dd0e6512062f4cc76cfa4287770974bd170af7e3a9528fa88e8` | 0.019552697520703068 |
| cae-hazelnut | `34ab16490dc5e75003971a54e53bbda61b2f6be9428f63ace322b06c64d7e811` | 0.0084340635221451452 |
| cae-transistor | `7849dab459cf2ae0c2a6dd764a2864998a298a6ad0e5498d6ee5882326cc0a2d` | 0.027364758774638176 |
| patchcore-bottle | `dcf0f8df3dc9b4be9ee1bcb897d3ec6f24a3191240dcfd4e68afd598acd77cde` | 31.620044136047362 |
| patchcore-hazelnut | `7b4c3784434d69724e8e7fb74d587e4c8a059faa24819dd556c224d03e580a93` | 48.837368011474609 |
| patchcore-transistor | `7917810678f656324236a0df6db639d7de93176a7a8ae9ef06e72f3bf0a57191` | 40.958555984497067 |

SHA-256 du registre de seuils : `f3128a255482f55cb416312780bc4832f4bfe24b017c9a2036089128ade68b54`.
Règle figée : `score > threshold` ⇒ suspect ; égalité ⇒ normal.
Les configs et seuils sont ancrés dans [configuration-hashes.json](configuration-hashes.json).

| Banque mémoire | Dimensions | SHA-256 des octets du tenseur CPU contigu |
|---|---|---|
| patchcore-bottle | [1710, 1536] | `be28dc2a4775507ba5a46a201d34b7e5d0a5a1f1a1258092d8fa91228f3656a2` |
| patchcore-hazelnut | [3205, 1536] | `2b7cc437fbff34fe137be8b5c19836ae93b92615108d38f7daaa9022e63d9bec` |
| patchcore-transistor | [1740, 1536] | `71a3406efc8fd24a1060c0163a87516e98a67ef638b19e82a891120167402355` |

## Garde-fous Git

GitHub n'a ni protection/règle de branche, ni contrôle CI configuré au moment de
l'audit ; le statut agrégé `pending` sans aucune vérification n'est pas un job
en cours. Les contrôles locaux ci-dessus servent de condition de promotion.
Le scan heuristique des fichiers suivis et nouveaux n'a trouvé aucun motif de
secret ; il ne constitue pas une garantie exhaustive. Aucun secret, dataset,
checkpoint ni environnement n'est ajouté au commit. Les fichiers lourds restent
exclus par `.gitignore`. Instantané : [safeguards.json](safeguards.json).

La promotion exige une nouvelle lecture des têtes distantes, un merge sans conflit,
une PR traçable et l'absence de contrôle GitHub en échec. Aucun push forcé et aucun
contournement de règle de branche ne sont autorisés par cette procédure.
