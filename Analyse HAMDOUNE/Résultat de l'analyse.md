<img src="
<img width="738" height="689" alt="image" src="https://github.com/user-attachments/assets/5ccc47f5-b199-4f9b-8343-5ae7b0fc93ec" />
" style="height:464px;margin-right:432px"/> 
<img src="<img width="629" height="635" alt="image" src="https://github.com/user-attachments/assets/d5336512-8ad3-46d0-ac97-6d1ef7df733d" />" style="height:464px;margin-right:432px"/> 


## OUMAIMA HAMDOUNE 
## Sommaire
Introduction

Présentation du Dataset

Méthodologie

Nettoyage des Données

Analyse Exploratoire

Matrice de Corrélation

Modélisation : Analyse Prédictive

Résultats et Interprétations

Conclusions et Recommandations
## 1. Introduction 
## 1.1. Origine et Contexte
La base de données Maladie Observations représente une collection de données médicales simulées conçue pour la recherche en intelligence artificielle médicale. Elle a été créée pour étudier comment les paramètres physiologiques fondamentaux — température, pouls, oxygénation, glycémie et tension — permettent de discriminer l'état de santé des patients. Cette approche s'inscrit dans la tendance de la médecine prédictive visant à identifier précocement les états pathologiques à partir de signes vitaux facilement mesurables.
## 1.2. Objectif Principal
L'objectif central de cette base est d'établir des relations quantifiables entre les mesures physiologiques routinières et le statut clinique des patients, permettant ainsi de développer des algorithmes d'aide au diagnostic capables de classer automatiquement les patients comme "sains" ou "malades" sur la base de critères objectifs
## 2. PRÉSENTATION DU DATASET
## 2.1. Description Générale
Type de données : Observations médicales de patients
Nombre d'observations : 1 500 entrées
Nombre de variables : 6 colonnes
## 2.2. Source des Données
# https://www.kaggle.com/datasets/nassimsfaxi/observation-de-maladie
## 2.3. variables 
## 2.3.1. Variables Mesurées
temperature - Régulation thermique corporelle (°C)

pouls - Activité cardiaque (battements/minute)

oxygene - Fonction respiratoire (% de saturation)

glycemie - Métabolisme glucidique (mg/dL)

tension - Hémodynamique (mmHg)

## 2.3.2.Variable Cible
label - Diagnostic binaire :

0 : État physiologique normal

1 : État pathologique détecté
## 3. Méthodologie 
Données Médicales Brutes
    ↓
Analyse des Données Manquantes
    ↓
Imputation par la Médiane
    ↓
Analyse des Distributions
    ↓
Détection des Outliers
    ↓
Analyse de Corrélation
    ↓
Analyse de la Variable Cible
    ↓
Interprétation Médicale
    ↓
Recommandations Cliniques
## 4. Outils et Technologies
Langage : Python 3.x

Bibliothèques :

pandas : manipulation de données

numpy : calculs numériques

matplotlib & seaborn : visualisations

scikit-learn : modélisation machine learning
## 5. Importation du DATA Base
```
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ignoring warning
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('/content/maladie_observations.csv')
display(df.head())
```
## 6. Nettoyage des données 
## 6.1. Checking for Duplicates and null values
```
df.duplicated().sum()
```
```
df.isnull().sum()
```
<img src="<img width="327" height="340" alt="image" src="https://github.com/user-attachments/assets/7e955737-358a-49d1-bf2e-9533aae11f0b" />" style="height:464px;margin-right:432px"/>
## 6.2.  Statistiques descriptives 
```
df.describe()
```
<img src="<img width="685" height="305" alt="image" src="https://github.com/user-attachments/assets/9f1147b9-5371-42e1-8872-81d0b595c382" />
" style="height:464px;margin-right:432px"/> 
## 6.3. problème détecté 
Après le nettoyage des données (suppression des doublons et imputation des valeurs manquantes), le principal problème détecté dans la base est la présence de valeurs aberrantes (outliers) dans plusieurs colonnes numériques clés.
Plus précisément :
## Température et Pouls présentent des valeurs significativement plus élevées que la majorité des données. Ces outliers pourraient indiquer des erreurs de mesure ou des cas médicaux extrêmes.
## Oxygène a également des outliers, avec des valeurs très faibles ou très élevées. Les valeurs très faibles, en particulier, méritent une attention particulière car elles peuvent avoir une signification clinique importante.
## Tension montre quelques outliers, principalement des valeurs élevées, bien que moins extrêmes que pour la température ou le pouls.
Ces outliers pourraient influencer l'entraînement de certains modèles d'apprentissage automatique et nécessitent souvent une investigation ou un traitement supplémentaire (par exemple, la suppression, la transformation, ou l'utilisation de méthodes robustes) en fonction des objectifs de l'analyse.
## 6.4. Imputer les Valeurs Manquantes

Remplir les valeurs manquantes dans les colonnes 'temperature', 'pouls' et 'oxygene' en utilisant la médiane de chaque colonne pour assurer la cohérence des données avant l'analyse graphique.

Je vais d’abord calculer la médiane pour chacune des colonnes spécifiées : temperature, pouls et oxygene. Ensuite, j’utiliserai ces valeurs médianes pour remplacer les valeurs manquantes (NaN) dans leurs colonnes respectives au sein du DataFrame df. Enfin, je vérifierai qu’il ne reste plus de valeurs manquantes dans ces colonnes.
```
median_temperature = df['temperature'].median()
median_pouls = df['pouls'].median()
median_oxygene = df['oxygene'].median()

df['temperature'].fillna(median_temperature, inplace=True)
df['pouls'].fillna(median_pouls, inplace=True)
df['oxygene'].fillna(median_oxygene, inplace=True)

print("Missing values after imputation:")
print(df[['temperature', 'pouls', 'oxygene']].isnull().sum())
```
<img src="
<img width="731" height="104" alt="image" src="https://github.com/user-attachments/assets/70e7dbe8-1837-4b20-8c65-99c1d36454c4" />
" style="height:464px;margin-right:432px"/> 
## 7. Résultats et analyses 
## 7.1. les Distributions et les outliers des variables 
## 7.1.1.la Température
## Distriution
```
plt.figure(figsize=(10, 6))
sns.histplot(df['temperature'], kde=True)
plt.title('Distribution de la Température')
plt.xlabel('Température')
plt.ylabel('Fréquence')
plt.show()
```
<img width="851" height="548" alt="image" src="https://github.com/user-attachments/assets/cea95972-6165-4d07-bc0d-ef39039809ff" />
La distribution montre que la grande majorité des températures sont normales et regroupées dans une plage réaliste (environ 35–40°C). Cependant, on observe des valeurs extrêmement élevées, allant jusqu’à plus de 500°C, ce qui est impossible en pratique et indique la présence de valeurs aberrantes ou d’erreurs de saisie.

Ces valeurs extrêmes étirent fortement l’axe horizontal, rendant la distribution difficile à lire et pouvant fausser les analyses statistiques.
Avant toute modélisation, il est indispensable de nettoyer ou corriger ces outliers afin d’obtenir une distribution plus cohérente et représentative des données réelles.
## les outiliers 
```
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['temperature'])
plt.title('Box Plot de la Température')
plt.ylabel('Température')
plt.show()
```
<img width="696" height="509" alt="image" src="https://github.com/user-attachments/assets/4c764307-8c9f-46cb-b294-e28806059944" />

Le boxplot montre que la majorité des températures sont normales, autour de 35–40°C. Cependant, de très nombreuses valeurs extrêmement élevées (jusqu’à plus de 500°C) apparaissent comme outliers. Ces valeurs sont impossibles d’un point de vue physiologique et correspondent donc à des erreurs de saisie. La présence de ces anomalies rend indispensable un nettoyage des données avant toute analyse fiable.
## 7.1.2. Pouls
## Distriution
```
plt.figure(figsize=(10, 6))
sns.histplot(df['pouls'], kde=True)
plt.title('Distribution du Pouls')
plt.xlabel('Pouls')
plt.ylabel('Fréquence')
plt.show()
```
<img width="851" height="547" alt="image" src="https://github.com/user-attachments/assets/2c1d529e-2ae4-4b43-a5d6-445e21d1e8a2" />
La distribution montre que la plupart des valeurs de pouls se situent dans une plage réaliste (environ 60 à 120 battements par minute). Cependant, comme pour la température, on observe des valeurs extrêmement élevées (jusqu’à près de 600), ce qui est physiologiquement impossible et révèle la présence d’outliers ou d’erreurs de saisie.

La forme étirée de la courbe vers la droite indique que ces valeurs aberrantes influencent fortement l’échelle du graphique. Un nettoyage des données est donc nécessaire avant toute analyse fiable.
## les outiliers 
```
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['pouls'])
plt.title('Box Plot du Pouls')
plt.ylabel('Pouls')
plt.show()
```
<img width="695" height="509" alt="image" src="https://github.com/user-attachments/assets/60c0e356-2ef9-4dea-a185-4acabbd4545d" />
Le boxplot montre que la majorité des valeurs de pouls se situent dans une plage normale (environ 60 à 120 bpm). Cependant, un grand nombre de points très éloignés au-dessus de cette plage apparaissent comme des outliers extrêmes, certains dépassant 300, 400 ou même 500 bpm — des valeurs impossibles physiologiquement.

Ces anomalies représentent des erreurs de mesure ou de saisie et faussent la distribution. Un nettoyage des données est donc indispensable avant toute analyse fiable ou modélisation.
## 7.1.3. l'Oxygène
## Distriution
```
plt.figure(figsize=(10, 6))
sns.histplot(df['oxygene'], kde=True)
plt.title("Distribution de l'Oxygène")
plt.xlabel('Oxygène')
plt.ylabel('Fréquence')
plt.show()
```
<img width="851" height="548" alt="image" src="https://github.com/user-attachments/assets/873d27cc-cc06-41d7-8517-2be6be80b1ec" />
La distribution montre que la grande majorité des valeurs d’oxygène se situent dans une plage réaliste (autour de 90–100 %). Cependant, comme pour les variables précédentes, on observe des valeurs extrêmement élevées, allant jusqu’à plus de 600 %, ce qui est impossible physiologiquement.

Ces valeurs aberrantes indiquent des erreurs de saisie ou d’enregistrement et étirent fortement la distribution, rendant la visualisation moins représentative. Un nettoyage de ces outliers est donc nécessaire pour obtenir une analyse fiable et cohérente.
## les outiliers 
```
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['oxygene'])
plt.title("Box Plot de l'Oxygène")
plt.ylabel('Oxygène')
plt.show()
```
<img width="696" height="509" alt="image" src="https://github.com/user-attachments/assets/89139457-5eb2-42b4-84a0-2021902e8a6c" />
Le boxplot montre que la majorité des valeurs d’oxygène sont regroupées autour d’une plage normale (environ 90–100 %). Toutefois, plusieurs valeurs extrêmement élevées apparaissent au-dessus du boxplot, atteignant parfois plus de 500 % ou 600 %, ce qui est totalement impossible d’un point de vue physiologique.

Ces valeurs correspondent à des outliers majeurs, probablement dus à des erreurs de saisie ou à un problème lors de la collecte des données. Elles faussent la représentation statistique et doivent être corrigées ou supprimées avant toute analyse fiable ou modélisation.
## 7.1.4.la Glycémie
## Distriution
```plt.figure(figsize=(10, 6))
sns.histplot(df['glycemie'], kde=True)
plt.title('Distribution de la Glycémie')
plt.xlabel('Glycémie')
plt.ylabel('Fréquence')
plt.show()
```
<img width="851" height="548" alt="image" src="https://github.com/user-attachments/assets/5cdb3545-da44-432b-b08a-712d666f0402" />
La distribution de la glycémie est globalement stable et régulière, avec des valeurs comprises entre environ 70 et 120 mg/dL, ce qui correspond à des niveaux réalistes et physiologiquement possibles. Contrairement aux autres variables (température, pouls, oxygène), aucune valeur extrême ou aberrante n’est visible.

La courbe de densité confirme une répartition relativement homogène, sans anomalies majeures. Cela indique que la glycémie est une variable propre, cohérente et directement exploitable pour l’analyse statistique ou la modélisation.
## les outiliers 
```
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['glycemie'])
plt.title('Box Plot de la Glycémie')
plt.ylabel('Glycémie')
plt.show()
```
<img width="696" height="509" alt="image" src="https://github.com/user-attachments/assets/bfa55834-0c2b-47d5-8a2f-40fc85ff6d88" />
Le boxplot montre que les valeurs de glycémie sont bien réparties entre environ 70 et 120 mg/dL, sans aucun outlier ni valeur aberrante. La médiane est autour de 94–95 mg/dL, et les quartiles indiquent une dispersion régulière des données.

Cette visualisation confirme que la glycémie est une variable propre, cohérente et fiable, ne nécessitant aucun nettoyage particulier. Elle peut donc être utilisée directement pour l’analyse ou la modélisation.
## 7.1.5.la Tension
## Distriution
```
plt.figure(figsize=(10, 6))
sns.histplot(df['tension'], kde=True)
plt.title('Distribution de la Tension')
plt.xlabel('Tension')
plt.ylabel('Fréquence')
plt.show()
```
<img width="851" height="547" alt="image" src="https://github.com/user-attachments/assets/bd3c5fe7-ea1d-42de-bc71-8bdd3e6a775b" />
La distribution de la tension montre une répartition homogène des valeurs, comprises entre environ 90 et 140 mmHg, ce qui correspond à une plage réaliste et physiologiquement cohérente. Aucun outlier extrême ou valeur aberrante n’est visible.

La courbe de densité est relativement régulière, indiquant une variabilité normale de la tension dans l’échantillon. Cette variable est donc propre, stable et directement exploitable pour l'analyse statistique ou la modélisation, sans besoin de nettoyage particulier.
## les outiliers
```
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['tension'])
plt.title('Box Plot de la Tension')
plt.ylabel('Tension')
plt.show()
```
<img width="695" height="509" alt="image" src="https://github.com/user-attachments/assets/4caa5078-3d13-4df4-851a-1007d8d4ee08" />
Le boxplot montre que les valeurs de tension sont bien réparties entre environ 90 et 140 mmHg, sans aucun outlier. La médiane se situe autour de 115 mmHg, ce qui correspond à une tension normale. Les quartiles indiquent une dispersion régulière et cohérente des données.

Ce graphique confirme que la tension est une variable propre, stable et exempte de valeurs aberrantes, ce qui la rend parfaitement exploitable pour l’analyse ou la modélisation sans nécessiter de nettoyage préalable.
## 7.2. Matrice des corrélations 
    
  



