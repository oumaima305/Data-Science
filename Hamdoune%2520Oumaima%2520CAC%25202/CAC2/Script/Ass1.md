
<img src="photo.jpg" style="height:464px;margin-right:432px"/>

## HAMDOUNE Oumaima

## N°2401031
## la base de données “Breast Cancer Wisconsin (Diagnostic)”                                                                                   ## contexte scientifique                                                                                                                      Le cancer du sein est l’une des principales causes de mortalité chez les femmes à travers le monde. Le diagnostic précoce joue un rôle déterminant dans le pronostic et le traitement.
L’analyse cytologique par aspiration à l’aiguille fine (Fine Needle Aspiration – FNA) est une méthode peu invasive permettant d’extraire des cellules d’une masse mammaire pour examen microscopique.
Toutefois, le diagnostic visuel dépend fortement de l’expérience du pathologiste et peut donc comporter une part de subjectivité.

C’est dans ce contexte que des chercheurs de l’Université du Wisconsin (Madison, USA) — le Dr. William H. Wolberg, W. Nick Street et Olvi L. Mangasarian — ont entrepris, en 1995, le développement d’un système de diagnostic assisté par ordinateur.
Leur objectif était de permettre la classification automatique des échantillons cellulaires en deux catégories :

B (Benign) : tumeur bénigne

M (Malignant) : tumeur maligne

Les résultats de leurs travaux ont été publiés dans plusieurs revues scientifiques, notamment :

“Nuclear feature extraction for breast tumor diagnosis” (Street et al., 1993)

“Breast cancer diagnosis and prognosis via linear programming” (Mangasarian et al., 1995)

“Computerized breast cancer diagnosis and prognosis from fine needle aspirates” (Wolberg et al., 1995)

Ces études pionnières ont montré que les caractéristiques morphologiques des noyaux cellulaires pouvaient être mesurées, quantifiées et utilisées pour prédire la nature de la tumeur avec un haut degré de précision.                                                         
## Objectif scientifique et applications                                                                                                 
L’objectif initial était de valider l’utilisation de mesures quantitatives pour remplacer partiellement l’observation visuelle du médecin.
Les résultats ont prouvé que :

Les caractéristiques morphologiques permettent de distinguer les deux types de tumeurs avec une précision supérieure à 95 % en utilisant des modèles linéaires.

Ce jeu de données peut servir de base d’entraînement pour des algorithmes modernes de machine learning (régression logistique, SVM, réseaux neuronaux, etc.).

Il est encore utilisé aujourd’hui comme jeu de référence dans la recherche et l’enseignement pour la classification binaire médicale.   
## Description du jeu de données  
La base de données Wisconsin Diagnostic Breast Cancer (WDBC) est l’une des plus connues du UCI Machine Learning Repository.
Elle contient 569 observations issues de prélèvements de patientes atteintes de tumeurs mammaires.

Chaque observation représente les mesures numériques d’un échantillon cellulaire obtenu par FNA.
Deux variables descriptives principales sont présentes :

id : identifiant unique du patient,

diagnosis : résultat du diagnostic (M = Malignant, B = Benign).

Les 30 autres variables sont des caractéristiques numériques calculées à partir d’images digitalisées de noyaux cellulaires.
Elles décrivent :

la taille,

la forme,

la texture,

la régularité et

la complexité des contours des noyaux.

Ces variables sont regroupées en trois catégories statistiques :

mean : moyenne de la mesure sur tous les noyaux observés.

se : erreur standard (Standard Error) de la mesure.

worst : plus mauvaise (ou plus grande) valeur mesurée.
## résultats et interprétation 

