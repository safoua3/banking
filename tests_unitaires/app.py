from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Supposons que df soit votre DataFrame chargé globalement
df = pd.read_csv('testing.csv')  # Chargez votre fichier de données ici

@app.route('/predict', methods=['GET'])
def predict():
    id = request.args.get('id', type=int)
    if id is None:
        return jsonify({"error": "ID manquant"}), 400

    print(f"ID reçu: {id}")
    print("Colonnes disponibles dans le DataFrame:", df.columns)

    if 'SK_ID_CURR' not in df.columns:
        return jsonify({"error": "La colonne 'SK_ID_CURR' n'existe pas dans le DataFrame."}), 500

    try:
        sample = df.loc[df['SK_ID_CURR'] == id]
        if sample.empty:
            return jsonify({"error": "ID non trouvé"}), 404
        
        # Ajoutez ici votre logique pour la prédiction
        prediction = "votre_prédiction"

        return jsonify({"id": id, "prediction": prediction})

    except KeyError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
