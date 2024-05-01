from flask import Flask, jsonify, request
import os
import sys
#import joblib
import pickle
import pandas as pd
import pytest
#from api import app,predict,model
directory=os.path.dirname(os.path.abspath(__file__))
# Add the directory of the file you want to import
#file_dir = 'directory/MODEL_PREDICTION/api.py'
#file_api=sys.path.append(os.path.abspath(file_dir))
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(directory,".."))
from app import app,predict,model

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture
# test du chargement du fichier csv de motre data
def test_data():
    # Determine le chemin du fichier CSV
    #path = os.path.join(current_directory, "..","test_data.csv")
    data = pd.read_csv('test_data.csv')
    # Vérifie que le DataFrame n'est pas vide
    assert not data.empty, "Error chargement du data."
    # Teste le chargement du modèle de prédiction
def test_model():
    # Détermine le chemin du fichier contenant le modèle entraîné
    #model_path = os.path.join(current_directory, "..", "model.pkl")
    # Charge le modèle à partir du fichier
    #model = joblib.load(model_path)
    model = pickle.load(open("model.pkl","rb"))

    # Vérifie que le modèle a été chargé correctement
    assert model is not None, "Error chargement du modèle."
    

def test_predict_1(client, test_data):
    response = client.get('/predict?id=100001')
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'probabilite' in response_data
    assert 'Pret' in response_data
    assert response_data['Pret'] in ['Accepté', 'Refusé']  # Check that the decision is one of the expected values

def test_predict_invalid(client):
    response = client.get('/predict?id=100002')
    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data == "cet identifiant n'existe pas"

def test_predict_incorrect(client):
    response = client.get('/predict')
    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data == "cet identifiant n'existe pas"
