import evidently
import pandas as pd
import numpy as np
import requests
from sklearn import datasets, ensemble
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

data=pd.read_csv('data_drift.csv')

#data_modelisation=data.dropna(subset=['TARGET']).drop(columns=['SK_ID_CURR','TARGET'])
#verifier que le data ne contient pas de valeurs manquantes sur la target et supprimer les colonnes sk_idd et target  qui ne sont pas pertinentes pour évaluer les performances du modèle.

data_modelisation = data.dropna(subset=['TARGET'])
data_modelisation= data_modelisation.drop(columns=['SK_ID_CURR','TARGET'])

#'TARGET' colonne représente la variable cible que le modèle tente de prédire.
# #En sélectionnant les lignes où la TARGET colonne est manquante, l' application_test ensemble de données
# #représente de nouvelles données que le modèle n'a pas vues auparavant 
# et peut être utilisé pour évaluer les performances du modèle sur de nouvelles données.
curent_data=data[data['TARGET'].isna()]
curent_data=curent_data.drop(columns=['SK_ID_CURR','TARGET'])

def print_column_names(data):
    print("Column names:")
    print(list(data.columns))

print_column_names(data_modelisation)
print_column_names(curent_data)

def align_column_names(reference_data, current_data):
    common_columns = reference_data.columns.intersection(current_data.columns)
    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]
    return reference_data, current_data

data_modelisation, curent_data = align_column_names(data_modelisation, curent_data)

#nous allons recuperer les colonnes categorielles deja encodees(one-hot-encoder)
#Pour les colonnes catégorielles binaires
categorical_columns = []

# Parcourir chaque colonne
for col in data_modelisation.columns:
    unique_vals = data_modelisation[col].nunique()
    if unique_vals <= 2 and data_modelisation[col].isin([0, 1]).all():
        categorical_columns.append(col)
        
# les colonnes numeriques sont les autres colonnes
numerical_columns = [col for col in data_modelisation.columns if col not in categorical_columns]


# Vérifiez que vos deux DataFrames ont exactement les mêmes colonnes
#assert set(data_modelisation.columns) == set(curent_data.columns)

# Si l'assertion est réussie, cela signifie que les colonnes correspondent
#print("Les colonnes correspondent!")

# Création du column mapping
column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_columns
column_mapping.categorical_features = categorical_columns

# Créer le rapport de dérive des données
dataDrift_report = Report(metrics=[DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

#print("le data driftreport")

dataDrift_report.run(reference_data=data_modelisation, current_data=curent_data, column_mapping=column_mapping)

#print("Run du data_drift_report")
dataDrift_report.show()

# Sauvegardez le rapport en tant que fichier HTML
dataDrift_report.save_html('dataDrift_report.html')
import webbrowser

url = 'dataDrift_report.html'
webbrowser.open(url)