from flask import Flask, jsonify, request
import os
import sys
import joblib
import pickle
import pandas as pd
import pytest


# Add the directory of the file you want to import
file_dir = '/MODEL_PREDICTION/app.py'
file_api=sys.path.append(os.path.abspath(file_dir))
from app import app,predict,model



# test du chargement du fichier csv de motre data
def test_chargement_data():
    # Détermine le chemin du fichier CSV
    #path = os.path.join(current_directory, "..","test_data.csv")
    data = pd.read_csv('test_data.csv')
    # Vérifie que le DataFrame n'est pas vide
    assert not data.empty, "Error chargement du data."
    
    
# Teste le chargement du modèle de prédiction
def test_chargement_model():
    # Détermine le chemin du fichier contenant le modèle entraîné
    #model_path = os.path.join(current_directory, "..", "model.pkl")
    # Charge le modèle à partir du fichier
    #model = joblib.load(model_path)
    model = pickle.load(open("model.pkl","rb"))

    # Vérifie que le modèle a été chargé correctement
    assert model is not None, "Error chargement du modèle."
    
# Teste la fonction de prédiction de l'API
def test_prediction():
    import os
    import pandas as pd
    from flask import json
    # Détermine le chemin du répertoire courant
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Détermine le chemin du fichier CSV contenant les données de test
    #csv_path = os.path.join(current_directory, "..", "Simulations", "Data", "df_train.csv")
    # Charge le fichier CSV dans un DataFrame pandas
    data = pd.read_csv('test_data.csv')
    # Prend un échantillon pour la prédiction
    id= data.iloc[2]['SK_ID_CURR']
    # Crée une requête de test pour la prédiction en utilisant l'échantillon sélectionné
    with app.test_client() as client:
        response = client.post('/predict', json={'SK_ID_CURR': id})
        df = json.loads(response.df)
        prediction = df['probability']
        # Vérifie que la prédiction a été effectuée correctement
        assert prediction is not None, "La prédiction a échoué."

