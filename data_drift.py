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

