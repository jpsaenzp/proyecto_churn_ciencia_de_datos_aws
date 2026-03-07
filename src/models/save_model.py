from pathlib import Path
import tensorflow as tf
import joblib

def save_best_model(models_dict, path="best_model"):
    """
    Identifica el mejor estimador dentro de un diccionario de modelos
    y lo guarda según su tipo.

    Parameters
    ----------
    models_dict : dict
        Diccionario de modelos entrenados o búsquedas de hiperparámetros
    path : str
        Ruta base donde guardar el modelo
    """

    best_model = None
    best_score = -1
    best_name = None

    for name, model in models_dict.items():

        # si es GridSearch / RandomSearch
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
            score = getattr(model, "best_score_", None)

        else:
            estimator = model
            score = getattr(model, "best_score_", None)

        if score is not None and score > best_score:
            best_score = score
            best_model = estimator
            best_name = name

    if best_model is None:
        raise ValueError("No se pudo identificar el mejor modelo")

    path = Path(path)

    # guardar red neuronal tensorflow
    if isinstance(best_model, tf.keras.Model):

        save_path = path.with_suffix(".keras")
        best_model.save(save_path)

    # guardar modelos sklearn/xgboost
    else:

        save_path = path.with_suffix(".joblib")
        joblib.dump(best_model, save_path)

    print(f"Best model: {best_name}")
    print(f"Saved at: {save_path}")

    return best_model, save_path