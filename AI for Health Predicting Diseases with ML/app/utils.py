import joblib

def load_model(model_path):
    """Loads a model from the specified path."""
    return joblib.load(model_path)

def preprocess_data(data, scaler_path):
    """Preprocesses input data using the scaler."""
    scaler = joblib.load(scaler_path)
    return scaler.transform(data)
