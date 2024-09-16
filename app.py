from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

# URI du modèle MLflow
model_uri = 'runs:/39da0c0dcad44e148a02727690a8979e/modele_regression_logistique'

# Chargement du modèle depuis MLflow
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([[data['credit_lines'], data['loan_amt'], data['total_debt'],
                                data['income'], data['years_employed'], data['fico_score']]])
        prediction = model.predict(input_data)

        return jsonify({
            'prediction': int(prediction[0]),
            'message': 'Prédiction effectuée avec succès'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)