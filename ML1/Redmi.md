
1) Importation du dataset
import sys
!{sys.executable} -m pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
import sys : importe le module système de Python.

# pip install ucimlrepo : installe la librairie permettant de télécharger des datasets UCI.
# fetch_ucirepo(id=186) : télécharge la base de données Wine Quality
# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

Ce code installe et utilise la librairie ucimlrepo pour télécharger automatiquement le dataset « Wine Quality ». Il affiche les métadonnées, les données ainsi que la description des variables afin de comprendre la structure du dataset avant toute analyse.
2) Construction du DataFrame
import pandas as pd
import numpy as np
# import pandas et numpy : librairies pour manipuler les données.
#pd.concat(...) : assemble les variables explicatives et la variable cible.
#df.shape : donne le nombre de lignes et colonnes.
# df.head() : affiche les premières lignes du dataset.
# The previous link resulted in a 404 error.
# We already have the full dataset (including both red and white wines)
# loaded via ucimlrepo into 'wine_quality.data.original'.
# Let's use that DataFrame directly.
df = wine_quality.data.original

print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())
On combine les variables explicatives et la variable cible dans un seul tableau df. Cela permet d’avoir toutes les informations regroupées dans un DataFrame adapté à l’analyse et au machine learning. L’affichage sert à vérifier que les données ont été correctement chargées.
3) Séparation entre X et Y
X = df.drop(["quality", "color"], axis=1) #we drop the column "quality" and "color"
Y = df["quality"]
print("\n========= Wine Qualities ========= \n")
print(Y.value_counts())
# f.drop : enlève les colonnes inutiles pour l’apprentissage.
#X : contient les variables explicatives.
#Y : contient la qualité du vin
#value_counts() : compte combien de vins appartiennent à chaque qualité.
Le code sépare les variables explicatives X et la variable cible Y. On affiche ensuite la distribution des classes pour vérifier l’équilibre du dataset. Cette étape prépare le jeu de données pour l’entraînement du modèle.
4) Transformation en classification binaire
 # bad wine (y=0) : quality <= 5 and good quality (y= 1) otherwise
 Y = [0 if val <=5 else 1 for val in Y]

On convertit la qualité du vin en deux classes : 0 pour les vins de mauvaise qualité et 1 pour les vins de bonne qualité. Cela simplifie la tâche de prédiction et permet d’utiliser un modèle de classification binaire.
5) Matrice de corrélation
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib et seaborn servent à tracer des graphiques.
#corr() : calcule les corrélations entre les variables.
#heatmap : affiche la matrice de corrélation.
plt.figure()
ax = plt.gca()
sns.boxplot(data=X,orient="v",palette="Set1",width=1.5, notch=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure()
# Exclude the 'color' column as it's non-numeric and causes the error
corr = X.drop('color', axis=1).corr()
sns.heatmap(corr)
Ce code calcule et affiche la matrice de corrélation des variables afin d’identifier les relations linéaires entre elles. Cela permet de repérer les variables influentes ou redondantes.
6) Division apprentissage / validation
from sklearn.model_selection import train_test_split
import numpy as np
#train_test_split : sépare les données en apprentissage et validation.
#test_size=0.5 : moitié apprentissage, moitié validation.
#stratify=Y : garde les mêmes proportions de classes.
# Ensure X only contains numerical features by dropping non-numeric columns
# This handles cases where 'color' or other non-numeric columns might persist
non_numeric_cols = X.select_dtypes(exclude=np.number).columns
if len(non_numeric_cols) > 0:
    print(f"Dropping non-numeric columns from X: {list(non_numeric_cols)}")
    X = X.drop(columns=non_numeric_cols)

Xa, Xt, Ya, Yt = train_test_split(X, Y, shuffle=True, test_size=1/3,
stratify=Y)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, shuffle=True, test_size=0.5,
stratify=Ya)
Les données sont divisées en deux ensembles : apprentissage et validation. L’option stratify garantit que les proportions des classes sont préservées, ce qui améliore la qualité de l’évaluation.
7) Premier modèle KNN
 from sklearn.neighbors import KNeighborsClassifier
 #KNN : algorithme basé sur les voisins les plus proches.
#fit() : entraîne le modèle.
#predict() : prédit les labels sur la validation.
#accuracy_score : calcule la précision.
 # Fit the model on (Xa, Ya)
 k = 3
 clf = KNeighborsClassifier(n_neighbors = k)
 clf.fit(Xa, Ya)
# Predict the labels of samples in Xv
 Ypred_v = clf.predict(Xv)
 # evaluate classification error rate
 from sklearn.metrics import accuracy_score
 error_v = 1-accuracy_score(Yv, Ypred_v)
Un premier modèle KNN est créé avec k=5 voisins. Il est entraîné puis testé sur les données de validation. L’erreur de classification est ensuite calculée pour mesurer la performance initiale du modèle.
8) Recherche du meilleur k
 # some hints
 #k_vector : liste des valeurs de k testées
k_vector = np.arange(1, 37, 2) #define a vector of k=1, 3, 5, ...
error_train = np.empty(k_vector.shape)
error_val = np.empty(k_vector.shape)
for ind, k in enumerate(k_vector):
    #fit with k
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(Xa, Ya)
    # predict and evaluate on training and validation sets
    Ypred_train = clf.predict(Xa)
    error_train[ind] = 1 - accuracy_score(Ya, Ypred_train)
    Ypred_val = clf.predict(Xv)
    error_val[ind] = 1 - accuracy_score(Yv, Ypred_val)
On teste plusieurs valeurs de k afin d’identifier celle donnant les meilleurs résultats. Pour chaque k, un nouveau modèle est entraîné et son erreur est enregistrée. Cela permet d’optimiser le modèle KNN.
9) Extraction du k optimal
#k_star = meilleur nombre de voisins.
 # some hints: get the min error and related k-value
 err_min, ind_opt = error_val.min(), error_val.argmin()
 k_star = k_vector[ind_opt]
On récupère la valeur de k qui génère l’erreur la plus faible. Cette valeur k_star représente le meilleur choix pour obtenir un modèle performant.
10) Normalisation des données
 #StandardScaler : normalisation des données.
#fit() : calcule les moyennes et écarts-types.
transform() : applique la normalisation sur les données.
 from sklearn.preprocessing import StandardScaler
 sc = StandardScaler(with_mean=True, with_std=True)
 sc = sc.fit(Xa)
 Xa_n = sc.transform(Xa)
 Xv_n = sc.transform(Xv)
La normalisation met toutes les variables numériques sur la même échelle, ce qui est indispensable pour le KNN. Cela empêche les variables avec de grandes valeurs de dominer le calcul des distances.
