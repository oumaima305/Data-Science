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
## 6.2.  Statistiques descriptives 
```
df.describe()
```
 
  
    

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }


  
    
      
      temperature
      pouls
      oxygene
      glycemie
      tension
      label
    
  
  
    
      count
      5706.000000
      5702.000000
      5715.000000
      5725.000000
      5725.000000
      5725.000000
    
    
      mean
      38.518146
      88.071773
      97.948465
      94.679671
      114.424629
      0.554410
    
    
      std
      14.472846
      30.605817
      21.843684
      14.388183
      14.483579
      0.497074
    
    
      min
      36.000615
      50.000000
      92.000184
      70.013976
      90.000000
      0.000000
    
    
      25%
      36.854110
      66.000000
      94.693446
      82.281469
      102.000000
      0.000000
    
    
      50%
      38.179327
      93.000000
      97.386048
      94.318157
      114.000000
      1.000000
    
    
      75%
      39.089746
      106.000000
      98.845349
      107.141442
      127.000000
      1.000000
    
    
      max
      522.520254
      591.064218
      597.940421
      119.984205
      139.000000
      1.000000
    
  


    

  
    

  
    
  
    

  
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  

    
      const buttonEl =
        document.querySelector('#df-da594509-b3aa-48e7-a937-be6f6e8a3463 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-da594509-b3aa-48e7-a937-be6f6e8a3463');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    
  


    
      


    
        
    

      


  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }


      
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-a04d5d84-5e38-4598-80a6-d958949ece44 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      
    

    
  



