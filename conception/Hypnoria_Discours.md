# Hypnoria — discours de soutenance

Texte à dire, slide par slide (identique aux notes du présentateur dans le PowerPoint). Environ 3054 mots, soit ≈ 23 minutes à un débit posé de 130 mots par minute, hors démonstration.

## 1. Titre

(≈ 30 s) Bonjour Mesdames et Messieurs les membres du jury. Je m'appelle Aymen Ftita, et j'ai le plaisir de vous présenter mon projet de fin d'études, réalisé au Centre National de l'Informatique, sous l'encadrement de M. Abdelouaheb Dahmeni et de Mme Hajer Dahmeni pour le CNI, et de Mme Ines Chouchane pour ESPRIT.

Ce projet s'appelle Hypnoria. C'est une plateforme qui classe automatiquement les stades du sommeil à partir d'un enregistrement polysomnographique, puis qui estime, en l'expliquant, la sévérité de l'apnée obstructive du sommeil.

## 2. Plan

(≈ 20 s) Ma présentation suit six temps. D'abord le contexte et la problématique, puis l'étude de l'existant. Ensuite la démarche et les données, avec deux extraits concrets du dataset. Puis les deux moteurs d'intelligence artificielle : la classification des stades du sommeil, puis la prédiction de la sévérité avec son explication, à chaque fois avec le code et les résultats. Je termine par la plateforme, le bilan et les perspectives.

## 3. Organisme d'accueil

(≈ 45 s) Le projet s'est déroulé du 26 janvier au 26 juillet 2026, au CNI à Tunis.

Le CNI est un établissement public créé en 1975 et certifié ISO 9001. C'est le principal appui technique des structures publiques tunisiennes pour leurs systèmes d'information, et il opère une infrastructure d'hébergement mutualisée. Mon projet s'inscrit dans ses axes « études et conseil » et « développement ».

Ce contexte a orienté trois choix techniques. La souveraineté des données : les métadonnées cliniques restent dans une base maîtrisée, les fichiers lourds dans un stockage objet accessible uniquement par des liens signés. La reproductibilité : toute la plateforme démarre en une commande grâce aux conteneurs. Et la traçabilité : chaque action sensible est journalisée.

## 4. Contexte clinique

(≈ 60 s) Côté médical, le diagnostic du sommeil repose sur la polysomnographie, l'examen de référence. Elle enregistre toute la nuit l'activité cérébrale par deux voies EEG, les mouvements des yeux par deux voies EOG, et le tonus musculaire du menton par l'EMG.

Un technicien découpe ensuite la nuit en époques de 30 secondes et attribue à chacune un stade selon les règles de l'AASM : éveil, N1, N2, N3 ou sommeil paradoxal, appelé REM. Une nuit normale contient environ 5 % de N1, la moitié de N2, et une vingtaine de pourcents de N3 et de REM.

Pour l'apnée obstructive du sommeil, la sévérité se mesure par l'index d'apnées-hypopnées, le nombre d'événements respiratoires par heure : moins de 5, c'est normal ; de 5 à 15, léger ; de 15 à 30, modéré ; au-delà de 30, sévère.

## 5. Problématique

(≈ 60 s) L'enjeu est considérable : selon Benjafield et al., environ 936 millions d'adultes seraient concernés dans le monde, dont 425 millions de formes modérées à sévères.

Or le diagnostic bute sur trois verrous. Le coût d'abord : une nuit de huit heures, c'est environ mille époques à classer à la main, donc plusieurs heures de travail expert par patient. La reproductibilité ensuite : deux experts ne s'accordent que dans environ 83 % des cas ; la référence elle-même est donc imparfaite. L'opacité enfin : les réseaux profonds atteignent ce niveau, mais restent des boîtes noires, et un médecin ne peut pas signer un diagnostic dont il ignore les motifs.

D'où notre problématique : comment produire automatiquement l'hypnogramme et la sévérité de l'AOS, de façon rapide, reproductible et explicable, dans un outil que le médecin peut réellement utiliser ?

## 6. Étude de l'existant

(≈ 60 s) J'ai étudié deux familles de solutions.

À gauche, les suites commerciales : Noxturnal, Sleepware, ProFusion, SleepWorks et EnsoSleep. Elles proposent surtout un scorage assisté. Elles sont liées au matériel de leur constructeur, vendues sous licence coûteuse, installées localement, et n'expliquent pas leurs décisions. En Tunisie, les laboratoires utilisent ces mêmes suites : il n'existe pas de solution nationale.

À droite, les travaux académiques. Depuis 2012, les modèles publiés atteignent 82 à 89 % de précision, c'est-à-dire le niveau de l'accord entre experts. Les deux meilleurs, SleepTransformer et XSleepNet, sont évalués sur SHHS, la cohorte que j'utilise aussi. Mais ces travaux s'arrêtent au modèle : pas d'outil clinique, pas d'estimation de la sévérité, pas d'explication.

## 7. Positionnement

(≈ 40 s) Ce tableau résume le positionnement d'Hypnoria face aux suites commerciales.

Notre valeur ajoutée tient dans la chaîne complète, à droite : un scorage automatique et indépendant du matériel, une estimation directe de la sévérité, une explication de chaque prédiction de sévérité, et un environnement web multi-utilisateurs avec dossier patient, collaboration entre médecins et traçabilité — sans coût de licence. C'est cet enchaînement, du signal brut jusqu'au rapport expliqué, qu'aucune des solutions étudiées ne couvre.

## 8. Démarche Scrum

(≈ 50 s) Pour conduire le projet, j'ai adopté Scrum, adapté à un développeur unique : les encadrants du CNI jouent le rôle de Product Owner, mon encadrante académique celui de Scrum Master.

Les sprints durent quatre semaines au lieu de deux, parce qu'un seul entraînement sur près de 200 000 exemples prend plusieurs jours. Le projet s'organise en trois releases de deux sprints : le moteur de classification, le diagnostic expliqué, puis la plateforme clinique.

Au total : 60 user stories et 358 points, tous livrés, avec une vélocité qui passe d'environ 38 à 80 points par sprint. Un risque s'est réalisé : l'accès aux données SHHS n'a été obtenu qu'en semaine 5. Le premier sprint a donc utilisé Sleep-EDF comme banc d'essai pour valider la chaîne de traitement.

## 9. Architecture de la solution

(≈ 50 s) Voici l'architecture fonctionnelle de la solution.

Le médecin dépose un fichier EDF. L'étape 1 découpe le signal en époques de 30 secondes ; trois réseaux — un CNN, un BiLSTM et un Transformer — combinés par un méta-apprenant produisent l'hypnogramme. L'étape 2 extrait de cet hypnogramme 25 marqueurs d'architecture du sommeil, les complète par des variables cliniques et d'oxymétrie, soit 53 variables, et prédit la classe de sévérité. L'étape 3 explique la prédiction avec SHAP et génère le rapport pour le médecin.

Le tout est servi par une plateforme web conteneurisée, que je présenterai en dernière partie.

## 10. Données

(≈ 50 s) Les données viennent de la Sleep Heart Health Study, une grande cohorte américaine distribuée par la National Sleep Research Resource.

Pour l'étape 1, j'ai utilisé 196 enregistrements de la première visite, soit 195 130 époques de 30 secondes, avec les cinq voies à 100 hertz. La distribution est très déséquilibrée : le stade N1 ne représente que 3,3 % des époques, d'où l'usage de poids de classe. Le découpage est 80/10/10, stratifié par époque — j'y reviendrai dans les limites.

Pour l'étape 2, j'utilise 2 647 patients de la deuxième visite, avec leur IAH mesuré, répartis en quatre classes de sévérité. 530 patients sont réservés au test.

## 11. Extrait du dataset — signal

(≈ 50 s) Voici concrètement à quoi ressemble une donnée de l'étape 1, extraite du sujet SHHS1-200001.

En haut, l'hypnogramme établi par l'expert pour toute la nuit : 1 084 époques de 30 secondes. En bas, l'époque 563, notée N2 par l'expert, sur les cinq voies : les deux EEG, les deux EOG et l'EMG du menton.

Le prétraitement est volontairement simple : lecture du fichier EDF et des annotations XML, rééchantillonnage à 100 hertz, découpage en fenêtres de 3 000 échantillons, puis normalisation z-score sur chaque fenêtre et chaque voie — ce qui évite toute fuite d'information par des statistiques globales. Chaque époque devient un tenseur de forme 5 par 3000, et chaque nuit est stockée dans un fichier d'environ 60 mégaoctets, pour que l'entraînement charge les données instantanément.

## 12. Extrait du dataset — variables

(≈ 45 s) Et voici un extrait réel de la matrice de l'étape 2 : une ligne par patient de SHHS2.

On y trouve d'abord les marqueurs calculés à partir de l'hypnogramme : temps de sommeil total, efficacité, part de sommeil profond N3, index de fragmentation. Puis les variables cliniques : saturation minimale en oxygène, index d'éveil, IMC, âge. Et enfin la cible : l'IAH mesuré et sa classe — ici en anglais, tel que dans le fichier.

Au total : 25 marqueurs d'architecture, 20 variables cliniques et 8 variables d'interaction, soit 53 variables. Point important : les indices de perturbation respiratoire ont été volontairement exclus. Ils mesurent directement les apnées : les garder reviendrait à donner la réponse au modèle.

## 13. Trois architectures

(≈ 60 s) Pour l'étape 1, j'ai comparé trois architectures qui lisent le signal différemment.

Le CNN multi-échelle combine une branche fine de 0,5 seconde, qui capte les fuseaux et les ondes rapides, et une branche large de 2 secondes pour les ondes lentes : c'est l'idée de DeepSleepNet. Le BiLSTM parcourt les 3 000 pas de temps dans les deux sens, et un mécanisme d'attention pondère les instants importants. Le Transformer découpe l'époque en 60 patchs de 50 échantillons : l'auto-attention porte alors sur 61 jetons au lieu de 3 000, ce qui la rend calculable.

Les trois sont entraînés de la même façon : AdamW, un taux d'apprentissage de 5·10⁻⁴ avec décroissance cosinus, 30 époques, une perte pondérée par l'inverse des fréquences de classe, et on garde le modèle qui maximise le kappa de validation. Avec deux montages et deux granularités, cela fait 12 modèles.

## 14. Code : BiLSTM avec attention

(≈ 40 s) Voici le code du BiLSTM, tel qu'il tourne en production dans le backend.

Le tenseur d'entrée, de forme batch × 5 × 3000, est transposé pour que le temps devienne la dimension de séquence. Le LSTM bidirectionnel à deux couches de 128 unités produit 256 valeurs à chaque instant. La couche d'attention calcule un poids par instant, normalisé par un softmax sur les 3 000 pas ; la somme pondérée donne un vecteur de contexte de 256 valeurs. Deux couches denses, avec dropout, produisent enfin les logits des classes.

## 15. Code : CNN multi-échelle

(≈ 40 s) Le CNN : les deux branches lisent le même signal en parallèle. La première avec un noyau de 50 échantillons, soit une demi-seconde ; la seconde avec un noyau de 200, soit deux secondes.

Chacune enchaîne convolution, normalisation, ReLU et pooling, puis un pooling adaptatif ramène la sortie à 16 positions, soit 2 048 valeurs par branche. Les deux sont concaténées en 4 096 valeurs, puis trois couches denses, avec dropout, classifient l'époque. C'est ce modèle qui obtient les meilleurs résultats.

## 16. Code : Transformer à patchs

(≈ 40 s) Le Transformer : la méthode unfold découpe l'époque en 60 patchs de 50 échantillons sur les cinq voies, soit 250 valeurs par patch, projetées en dimension 128.

On ajoute en tête un jeton CLS appris, puis un encodage positionnel sinusoïdal. L'encodeur, quatre couches à huit têtes, fait interagir les 61 jetons entre eux, et la classification se fait sur le seul jeton CLS, qui résume toute l'époque.

## 17. Résultats des 12 modèles

(≈ 60 s) Voici les résultats des 12 modèles sur le jeu de test.

Trois enseignements. Premièrement, le CNN est le meilleur dans les quatre configurations : notre meilleur modèle unique atteint 91,66 % de précision et un kappa de 0,853, en 5 voies et 3 classes. Deuxièmement, passer de 2 à 5 voies apporte entre 2,6 et 4,4 points : l'EOG et l'EMG aident vraiment. Troisièmement, passer de 3 à 5 classes coûte une dizaine de points, et c'est le stade N1 qui en est responsable — je le montre sur la slide suivante.

## 18. Ensembles et stade N1

(≈ 60 s) J'ai ensuite combiné les trois réseaux, soit par vote moyen, soit par stacking avec une régression logistique, évalués cette fois avec un découpage par sujet, sur 20 314 époques.

Le meilleur kappa atteint 0,950 en 5 voies et 3 classes. Le stacking gagne quelques millièmes sur le vote moyen dans trois configurations sur quatre. Attention cependant : ce protocole diffère de celui des modèles seuls, donc je ne présente pas l'écart avec 91,66 % comme un gain du stacking.

À droite, la vraie limite : le stade N1. Sur les 638 époques N1 du test, seules 150 sont reconnues ; les autres partent vers N2, l'éveil ou le REM. C'est un stade de transition, et c'est aussi celui sur lequel les experts humains divergent le plus.

## 19. Étape 2 — de l'hypnogramme à la sévérité

(≈ 50 s) L'étape 2 transforme l'hypnogramme en diagnostic.

On calcule 25 marqueurs AASM — latences, efficacité, proportions de stades, fragmentation, probabilités de transition — auxquels s'ajoutent 20 variables cliniques et d'oxymétrie, et 8 variables d'interaction : 53 variables au total.

Six classifieurs ont été comparés en validation croisée stratifiée à 5 plis, sur le F1 macro. La forêt aléatoire, XGBoost et LightGBM arrivent en tête, avec des scores très proches. Ces trois modèles sont empilés sous une régression logistique équilibrée, et la sortie est l'une des quatre classes d'IAH.

## 20. Code XGBoost et stacking

(≈ 45 s) Voici le code d'entraînement de l'étape 2, extrait du script models-save.py.

XGBoost utilise 600 arbres de profondeur 6, avec un taux d'apprentissage de 0,05. Chaque arbre ne voit que 80 % des patients et 75 % des variables, et une régularisation L1 et L2 s'applique : c'est indispensable pour ne pas sur-apprendre sur un peu plus de 2 000 patients d'entraînement.

La sélection des modèles se fait par validation croisée stratifiée à 5 plis. Les trois meilleurs sont empilés : le méta-apprenant reçoit leurs probabilités calculées hors-pli, ce qui évite qu'il apprenne sur des prédictions trop optimistes. Enfin, en production, TreeSHAP est calculé sur le modèle XGBoost pour expliquer chaque patient.

## 21. Résultats de la sévérité

(≈ 60 s) Sur les 530 patients de test, la forêt aléatoire obtient 57 % de précision et un kappa de 0,41, soit un accord modéré. Surtout, 95,5 % des patients sont classés dans la bonne classe ou dans une classe voisine. Pour comparaison, prédire toujours la classe majoritaire ne donne que 36 %.

La matrice de confusion le confirme : les erreurs se font presque toujours entre classes adjacentes. Seuls deux patients normaux sont classés sévères, et aucun patient sévère n'est classé normal — ce qui est l'erreur la plus grave cliniquement.

Résultat honnête : le stacking fait un peu moins bien, 54,5 %, parce que ses trois modèles sont tous des ensembles d'arbres : ils se trompent sur les mêmes patients, il n'y a pas assez de diversité à combiner.

## 22. Explicabilité SHAP

(≈ 60 s) Pour l'explicabilité, j'utilise SHAP, qui répartit chaque prédiction entre les variables selon la valeur de Shapley, issue de la théorie des jeux.

À gauche, l'importance globale, calculée sur les 530 patients de test. L'oxymétrie domine : le pourcentage du temps passé sous 90 % de saturation, puis l'amplitude des désaturations. Viennent ensuite les indices d'éveil, l'IMC et l'âge. C'est cohérent avec ce que l'on sait cliniquement de l'apnée.

À droite, ce que voit le médecin pour un patient précis : ici un patient du test avec un IAH réel de 36,5, prédit sévère. Les éveils fréquents et l'interaction IMC-éveils poussent vers « sévère » ; à l'inverse, une saturation peu dégradée tire dans l'autre sens. Le médecin voit donc pourquoi le modèle conclut, et peut le contester. Une précision importante : SHAP décrit le modèle, pas une causalité physiologique.

## 23. Architecture de la plateforme

(≈ 45 s) Passons à la plateforme. Elle tient en trois conteneurs orchestrés par Docker Compose : un serveur nginx qui sert l'interface React 19, un backend FastAPI qui embarque les modèles PyTorch et scikit-learn, et une base PostgreSQL 15.

Les fichiers volumineux — les EDF, les hypnogrammes et les rapports — sont confiés à un stockage objet compatible S3, accessible uniquement par des URL signées valables 24 heures : c'est la réponse à la contrainte de souveraineté évoquée au début.

L'API expose 36 routes, réparties entre l'administration, les examens, l'inférence, la collaboration, les patients et l'authentification.

## 24. Interface médecin

(≈ 60 s, plus démonstration si le temps le permet) Voici l'interface.

À gauche, la page de résultats : l'hypnogramme prédit, superposé à la saturation en oxygène et à la densité d'événements sur un même axe temporel, puis les métriques AASM comparées aux normes, avec des alertes cliniques.

À droite, l'assistant d'analyse en quatre étapes, ici en thème sombre : choix du montage — 5 ou 2 voies —, choix de la granularité, dépôt du fichier, puis les résultats.

La plateforme gère aussi le dossier patient, l'annotation époque par époque, un rapport PDF dont on choisit les sections, les avis confraternels entre médecins, et une console d'administration avec gestion des licences, des hôpitaux et un journal d'audit exportable. L'interface est bilingue, français et anglais.

## 25. Industrialisation et sécurité

(≈ 45 s) Côté industrialisation, chaque push ou pull request sur la branche principale déclenche une intégration continue en trois jobs. Le backend est compilé, puis réellement démarré avec une base PostgreSQL, et l'on vérifie l'authentification et le cloisonnement des rôles. Le frontend est analysé puis construit. Enfin, les deux images de conteneurs sont construites.

Côté sécurité : les mots de passe sont hachés avec bcrypt, les jetons JWT sont signés et expirent après 7 jours, chaque requête vérifie le rôle et la propriété de la ressource, et les fichiers ne sont accessibles que par des liens signés.

Les 18 scénarios de test fonctionnels sont tous passés. Ce sont des tests manuels : leur automatisation fait partie des perspectives.

## 26. Bilan

(≈ 40 s) En bilan : sur les quatre objectifs fixés, trois sont atteints et le quatrième l'est partiellement.

Le scorage automatique est atteint, avec 91,66 % pour le meilleur modèle seul. La prédiction de la sévérité est atteinte, avec plus de 95 % des patients classés à une classe près, sans aucune annotation respiratoire manuelle. La chaîne applicative complète est livrée : les 60 user stories, soit 358 points.

L'explicabilité, en revanche, n'est que partielle : la sévérité est expliquée par SHAP, mais pas encore la classification des stades.

## 27. Limites et perspectives

(≈ 60 s) Je tiens à être transparent sur les limites.

Un : pour les modèles seuls, le découpage est fait par époque et non par sujet, ce qui peut rendre les scores optimistes. Deux : le corpus de 196 enregistrements ne représente que 3,4 % de la cohorte SHHS1. Trois : l'explicabilité ne couvre pas encore le staging, et SHAP n'a pas de valeur causale. Quatre : avant une mise en production, il faut authentifier les routes d'inférence, compléter la conformité RGPD et automatiser les tests.

Les perspectives répondent directement à ces limites : réévaluer par sujet et valider sur une cohorte externe, MESA, dont j'ai déjà obtenu l'accès ; sécuriser la plateforme et mener une étude de concordance clinique avec des médecins ; et étendre l'explicabilité au staging, avec les poids d'attention et des cartes de saillance.

## 28. Références

(≈ 10 s) Voici les principales références citées dans la présentation ; la bibliographie complète figure dans le rapport.

## 29. Merci

(≈ 15 s) Je vous remercie pour votre attention. Je suis à votre disposition pour répondre à vos questions, et je peux vous faire une courte démonstration de la plateforme.

