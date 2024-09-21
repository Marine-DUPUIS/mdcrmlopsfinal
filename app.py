from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

# Charger le modèle depuis un chemin local
model_uri = "models:/RL MLOPS/Production"
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du POST request
        data = request.json
        
        # Créer les données d'entrée pour le modèle à partir des données envoyées
        input_data = np.array([[data['credit_lines'], data['loan_amt'], data['total_debt'],
                                data['income'], data['years_employed'], data['fico_score']]])
        
        # Faire la prédiction avec le modèle chargé
        prediction = model.predict(input_data)

        # Retourner la prédiction dans une réponse JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'message': 'Prédiction effectuée avec succès'
        })
    except Exception as e:
        # En cas d'erreur, retourner un message d'erreur
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Lancer le serveur Flask sur le port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
