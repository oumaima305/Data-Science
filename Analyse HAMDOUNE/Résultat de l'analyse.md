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
## les outiliers 
## 7.1.3. l'Oxygène
## Distriution
## les outiliers 
## 7.1.4.la Glycémie
## Distriution
## les outiliers 
## 7.1.5.la Tension
## Distriution
## les outiliers 
## 7.2. Matrice des corrélations 
    
  



