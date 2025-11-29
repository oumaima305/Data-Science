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


    
  



