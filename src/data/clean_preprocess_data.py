import pandas as pd
import numpy as np
import csv
import re


def estandarizar_nulos(df, personalizar_nulos=None):
    """
    Reemplaza múltiples representaciones de null por np.nan
    """
    default_nulls = ["null", "Null", "NULL", "na", "NA", "n/a", "N/A", "nan", "NaN", "", " ", "  ", "-", "--"]
    if personalizar_nulos:
        default_nulls.extend(personalizar_nulos)
    # Reemplazo global
    df = df.replace(default_nulls, np.nan)
    # Strip en columnas string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(default_nulls, np.nan)
    return df


def analizar_fecha_mixta(date_str):
    """
    analiza formatos de fecha
    """
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    # dividir usando múltiples separadores
    parts = re.split(r"[\/\-.\\]", date_str)
    if len(parts) != 3:
        raise ValueError(f"Formato inesperado: {date_str}")
    nums = [int(p) for p in parts]
    # detectar año
    year = None
    for n in nums:
        if len(str(n)) == 4:
            year = n
    if year is None:
        raise ValueError(f"No se detectó año en {date_str}")
    # quitar año de la lista
    remaining = [n for n in nums if n != year]
    # detectar día
    day = None
    for n in remaining:
        if n > 12:
            day = n
    # si ninguno >12, asumir primer valor como mes (formato americano)
    if day is None:
        month = remaining[0]
        day = remaining[1]
    else:
        month = [n for n in remaining if n != day][0]
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalizar_columna_fecha(df, col):
    """
    normaliza formatos de fecha
    """
    df = df.copy()
    df[col] = df[col].apply(analizar_fecha_mixta)
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
    return df


def eliminar_duplicados(df, subset):
    """
    Elimina duplicados basados en 'subset' y conserva la fila con más valores válidos.
    """
    # Paso 1: eliminar duplicados exactos
    df = df.drop_duplicates()
    # Paso 2: dentro de cada grupo, conservar la fila con menos NaN
    df_limpio = (df.groupby(subset, group_keys=False).apply(lambda g: g.loc[g.isna().sum(axis=1).idxmin()]))
    return df_limpio.reset_index()


def detectar_outliers(df, features=None, metodo="IQR", umbral=1.5):
    """
    Detecta outliers en columnas numéricas y retorna un diccionario
    con índice, customer_id, feature y valor del outlier.
    """
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
    outliers = {}
    for col in features:
        serie = df[col].dropna()
        if metodo.upper() == "IQR":
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - umbral * iqr
            upper = q3 + umbral * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        elif metodo.upper() == "Z":
            mean = serie.mean()
            std = serie.std()
            mask = ((df[col] - mean).abs() > umbral * std)
        else:
            raise ValueError("Método debe ser 'IQR' o 'Z'")
        # Construir lista de diccionarios con índice, customer_id, feature y valor
        outliers[col] = [
            {"index": idx, "customer_id": df.loc[idx, "customer_id"], "feature": col, "value": df.loc[idx, col]}
            for idx in df[mask].index
        ]
    return outliers


def outliers_a_dataframe(outliers_dict):
    """
    Convierte el diccionario de outliers en un DataFrame.
    """
    lista = []
    for feature, registros in outliers_dict.items():
        for r in registros:
            lista.append(r)  # cada r ya tiene index, customer_id, feature, value
    return pd.DataFrame(lista)


def winsorize_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df[col] = df[col].clip(lower, upper)
    return df