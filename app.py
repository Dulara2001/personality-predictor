from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Global variables for loaded objects
model = None
scaler = None
target_encoder = None
label_encoders = None
feature_columns = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, target_encoder, label_encoders, feature_columns
    try:
        model = joblib.load('D:\\Decryptogen\\personality_model.pkl')
        scaler = joblib.load('D:\\Decryptogen\\scaler.pkl')
        target_encoder = joblib.load('D:\\Decryptogen\\target_encoder.pkl')
        label_encoders = joblib.load('D:\\Decryptogen\\label_encoders.pkl')
        feature_columns = joblib.load('D:\\Decryptogen\\feature_columns.pkl')
        print("All model components loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model when application starts
load_model()

@app.route('/')
def home():
    return jsonify({
        'message': 'Personality Prediction API',
        'status': 'active',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'features': '/features (GET)'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/predict',
            'content-type': 'application/json',
            'example_request': {
                'Time_spent_Alone': 4,
                'Stage_fear': 'No',
                'Social_event_attendance': 4,
                'Going_outside': 6,
                'Drained_after_socializing': 'No',
                'Friends_circle_size': 13,
                'Post_frequency': 5
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return the expected feature names and types"""
    if feature_columns is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get feature types from label_encoders
    feature_info = {}
    for col in feature_columns:
        if col in label_encoders:
            feature_info[col] = {
                'type': 'categorical',
                'allowed_values': label_encoders[col].classes_.tolist()
            }
        else:
            feature_info[col] = {
                'type': 'numerical',
                'description': 'Integer or float value'
            }
    
    return jsonify({
        'features': feature_columns,
        'feature_info': feature_info,
        'target_classes': target_encoder.classes_.tolist() if target_encoder else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded', 'status': 'failed'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided', 'status': 'failed'}), 400
        
        # Create input DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess categorical variables
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    # Handle categorical encoding
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}. Allowed values: {label_encoders[col].classes_.tolist()}',
                        'status': 'failed'
                    }), 400
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                return jsonify({
                    'error': f'Missing feature: {col}',
                    'status': 'failed',
                    'required_features': feature_columns
                }), 400
        
        # Reorder columns to match training
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Get probabilities with error handling
        try:
            prediction_proba = model.predict_proba(input_scaled)[0]
            prob_dict = {
                target_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(prediction_proba)
            }
        except AttributeError:
            # If model doesn't support predict_proba
            classes = target_encoder.classes_
            prob_dict = {
                cls: 1.0 if i == prediction else 0.0 
                for i, cls in enumerate(classes)
            }
        
        # Convert back to original label
        personality = target_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            'prediction': personality,
            'probabilities': prob_dict,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Endpoint to reload the model (useful for updates)"""
    if load_model():
        return jsonify({'message': 'Model reloaded successfully', 'status': 'success'})
    else:
        return jsonify({'error': 'Failed to reload model', 'status': 'failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)