
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
## Les codes 
import matplotlib.pyplot as plt
import seaborn as sns

# Get value counts for 'Diagnosis'
diagnosis_counts = y['Diagnosis'].value_counts()

# Create the bar plot
#sns.countplot() : crée un diagramme en barres montrant le nombre d’échantillons pour chaque type de diagnostic (B ou M)
plt.figure(figsize=(7, 5))
sns.countplot(x='Diagnosis', data=y, palette='viridis')

# Add labels and title
#palette='viridis' : palette de couleurs utilisée pour rendre le graphique lisible et esthétique
plt.title('Distribution of Breast Cancer Diagnoses')
plt.xlabel('Diagnosis Type')
plt.ylabel('Number of Cases')

# Display the plot
plt.show()
## **Code Python -IMPORTATION DU DATA :**
## la phase d’importation et d’exploration initiale des données


```python
!pip install ucimlrepo
import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

print("First 5 rows of features (X):\n", X.head())
print("\nFirst 5 rows of target (y):\n", y.head())
print("\nValue counts for 'Diagnosis' in y:\n", y['Diagnosis'].value_counts())
```
 Installer et utiliser la bibliothèque ucimlrepo pour importer des jeux de données depuis le UCI Machine Learning Repository,

Charger le dataset Breast Cancer Wisconsin (Diagnostic) (ID = 17 dans la base UCI),

Afficher les premières lignes des variables explicatives (features) et de la variable cible (target),

Observer la répartition des classes de diagnostic (bénin vs malin).
## **Code Python -VISUALISATION AVEC GRAPHES :**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get value counts for 'Diagnosis'
diagnosis_counts = y['Diagnosis'].value_counts()

# Create the bar plot
#sns.countplot() : crée un diagramme en barres montrant le nombre d’échantillons pour chaque type de diagnostic (B ou M)
plt.figure(figsize=(7, 5))
sns.countplot(x='Diagnosis', data=y, palette='viridis')

# Add labels and title
#palette='viridis' : palette de couleurs utilisée pour rendre le graphique lisible et esthétique
plt.title('Distribution of Breast Cancer Diagnoses')
plt.xlabel('Diagnosis Type')
plt.ylabel('Number of Cases')

# Display the plot
plt.show()
```
Ce code permet de visualiser la répartition des diagnostics de cancer du sein dans le jeu de données. À l’aide de seaborn.countplot(), il trace un diagramme en barres indiquant le nombre de cas bénins (B) et malins (M), avec une palette de couleurs « viridis » pour un rendu clair et esthétique. Le graphique révèle que les tumeurs bénignes sont plus fréquentes que les tumeurs malignes, ce qui montre un léger déséquilibre entre les classes. Cette visualisation simple mais essentielle aide à comprendre la composition du dataset avant d’appliquer des méthodes d’analyse ou de prédiction.
```python
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib.pyplot :  créer des graphiques et des figures 
# Get value counts for 'Diagnosis'
diagnosis_counts = y['Diagnosis'].value_counts()

# Create the bar plot, addressing the FutureWarning
# sns.countplot() :trace un diagramme en barres
plt.figure(figsize=(7, 5))
sns.countplot(x='Diagnosis', data=y, hue='Diagnosis', palette='viridis', legend=False)

# Add labels and title
plt.title('Distribution of Breast Cancer Diagnoses')
plt.xlabel('Diagnosis Type')
plt.ylabel('Number of Cases')

# Display the plot
plt.show()
# M = Malignant (tumeur maligne)
# B = Benign (tumeur bénigne)
```
Ce code a pour objectif de visualiser la répartition des diagnostics de cancer du sein dans le jeu de données. Après avoir compté le nombre de cas bénins (B) et malins (M) avec value_counts(), un diagramme en barres est tracé à l’aide de seaborn.countplot(). Le graphique montre que les tumeurs bénignes sont plus nombreuses que les tumeurs malignes, indiquant un léger déséquilibre dans le dataset. Cette étape permet de comprendre la composition des classes avant toute analyse prédictive et de vérifier si un rééquilibrage des données pourrait être nécessaire lors de la modélisation.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Define the features for plotting
features_to_plot = ['radius1', 'texture1', 'perimeter1']

# Create a figure with subplots for the box plots
plt.figure(figsize=(18, 6))

for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 3, i + 1) # 1 row, 3 columns, current plot index
    sns.boxplot(x='Diagnosis', y=feature, data=df_combined, palette={'M': 'salmon', 'B': 'lightgreen'})
    plt.title(f'Distribution of {feature} by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel(feature)
    plt.legend(title='Diagnosis', loc='upper right', labels=['Malignant', 'Benign'])

plt.tight_layout()
plt.show()
#radius1 → rayon moyen des noyaux cellulaires,
#texture1 → variation de l’intensité de la texture,
#perimeter1 → périmètre du contour des noyaux.
```
Ce code a pour objectif de comparer la distribution de trois caractéristiques cellulaires — radius1, texture1 et perimeter1 — selon le type de diagnostic (bénin ou malin). À l’aide de seaborn.boxplot(), il trace trois boxplots côte à côte, où les tumeurs malignes sont représentées en rouge saumon et les bénignes en vert clair. Ces graphiques permettent d’observer que les tumeurs malignes présentent généralement des valeurs plus élevées pour le rayon et le périmètre, traduisant des cellules plus grandes et irrégulières, tandis que les tumeurs bénignes ont des caractéristiques plus petites et régulières. Cette visualisation met donc en évidence des différences morphologiques importantes entre les deux types de tumeurs, utiles pour la classification.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Define the features for plotting
features_to_plot = ['radius1', 'texture1', 'perimeter1']

# Create a figure with subplots for the box plots
plt.figure(figsize=(18, 6))

for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 3, i + 1) # 1 row, 3 columns, current plot index
    sns.boxplot(x='Diagnosis', y=feature, hue='Diagnosis', data=df_combined, palette={'M': 'salmon', 'B': 'lightgreen'})
    plt.title(f'Distribution of {feature} by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
```
Ce code permet de visualiser la distribution de trois caractéristiques cellulaires — radius1, texture1 et perimeter1 — en fonction du type de diagnostic (bénin ou malin). À l’aide de seaborn.boxplot(), il trace trois boxplots côte à côte, colorés selon le diagnostic : rouge saumon pour les tumeurs malignes et vert clair pour les bénignes. Ces graphiques montrent que les valeurs du rayon et du périmètre sont en moyenne plus élevées pour les tumeurs malignes, ce qui indique que les cellules cancéreuses ont tendance à être plus grandes et irrégulières, tandis que les tumeurs bénignes présentent des valeurs plus faibles et stables. Cette visualisation aide à comprendre comment certaines caractéristiques morphologiques distinguent clairement les deux types de tumeurs.


