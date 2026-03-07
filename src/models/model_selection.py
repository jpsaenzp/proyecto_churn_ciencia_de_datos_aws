import pandas as pd

def seleccionar_mejor_modelo(models_dict):
    """
    Evalúa un diccionario de modelos entrenados y devuelve
    un DataFrame con los resultados ordenados y el mejor modelo.

    Parameters
    ----------
    models_dict : dict
        Diccionario con nombre del modelo como clave y objeto del modelo como valor.

    Returns
    -------
    df_resultados : pd.DataFrame
        DataFrame con los modelos y sus puntajes ordenados por best_score_.
    best_model : object
        El mejor modelo según best_score_.
    """
    results = {}
    for name, model in models_dict.items():
        if hasattr(model, "best_score_"):
            results[name] = model.best_score_
        else:
            results[name] = None  # por si algún modelo no tiene best_score_

    df_resultados = pd.Series(results).sort_values(ascending=False).reset_index()
    df_resultados.columns = ["Model", "Best_Score"]

    best_model = max(
        models_dict.items(),
        key=lambda x: getattr(x[1], "best_score_", -1)
    )[1]

    return df_resultados, best_model