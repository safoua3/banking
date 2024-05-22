#https://safoua-ea72e808f916.herokuapp.com/predict?id=
import os
import pickle
import pandas as pd
#import shap
from flask import Flask, jsonify, request

app = Flask(__name__)



# Charger le modèle en dehors de la clause if __name__ == "__main__":
#model_path = os.path.join(current_directory, "..", "Simulations", "Best_model", "model.pkl")
#model = pickle.load(open("model.pkl","rb"))
model = pickle.load(open("best_model.pkl","rb"))
@app.route("/predict", methods=['GET'])
def predict():
    id =int(request.args.get('id'))
    if id==None :
        return "cet identifiant n'existe pas"
     
    else:
        print(str(id))
        #df = pd.read_csv("test_data.csv")
        df = pd.read_csv("testing.csv")

        sample = df.loc[df['SK_ID_CURR']==id]
        #print(sample)
        sample = sample.drop(columns=['SK_ID_CURR'])
        #print(sample)
        proba = model.predict_proba(sample)[:, 1][0]
        #proba = prediction[0][1]
        print(proba)
        seuil=0.575
        if proba >= seuil:
            Pret = "Accepté"
        else:
            Pret = "Refusé"
        return jsonify({'probabilite': proba, 'Pret': Pret})
    
    


if __name__ == '__main__':
    app.run(debug=True,port=5002)

