from pathlib import Path
import joblib
import tensorflow as tf
import numpy as np

def get_predictions(model, X, threshold=0.5):
    """
    Devuelve predicciones binarias y probabilidades para modelos sklearn y Keras.
    
    Parameters
    ----------
    model : sklearn estimator o tf.keras.Model
        Modelo entrenado.
    X : array-like
        Datos de entrada.
    threshold : float
        Umbral para clasificación binaria.
    
    Returns
    -------
    y_pred : np.array
        Predicciones binarias (0/1).
    y_prob : np.array
        Probabilidades estimadas.
    """
    # Caso sklearn
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    # Caso Keras/TensorFlow
    else:
        y_prob = model.predict(X).flatten()
    
    y_pred = (y_prob > threshold).astype(int)
    return y_pred, y_prob


def predict_churn(model_dir, X, threshold=0.5):
    """
    Carga automáticamente el modelo dentro de una carpeta y realiza predicciones.

    Parameters
    ----------
    model_dir : str
        Carpeta donde está guardado el modelo
    X : dataframe o array
        Datos de entrada
    threshold : float
        Umbral de clasificación

    Returns
    -------
    prediction : np.array
    proba : np.array
    """

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    # buscar archivos de modelo soportados
    model_files = list(model_dir.glob("*.joblib")) + \
                  list(model_dir.glob("*.keras")) + \
                  list(model_dir.glob("*.h5"))

    if len(model_files) == 0:
        raise ValueError("No supported model file found in directory")

    # si hay más de un archivo, tomar el más reciente
    model_path = max(model_files, key=lambda f: f.stat().st_mtime)

    # cargar modelo según extensión
    if model_path.suffix == ".joblib":

        model = joblib.load(model_path)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X)

    elif model_path.suffix in [".keras", ".h5"]:

        model = tf.keras.models.load_model(model_path)
        proba = model.predict(X).flatten()

    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    prediction = (proba > threshold).astype(int)

    return prediction, proba