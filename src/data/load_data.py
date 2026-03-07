import pandas as pd
import csv

def cargar_csv_con_sniffer(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo CSV detectando automáticamente el delimitador.
    Si el DataFrame resultante tiene una sola columna, divide manualmente
    usando el delimitador detectado.
    
    Parámetros:
    -----------
    ruta_archivo : str
        Ruta del archivo CSV.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con los datos del archivo.
    """
    # Detectar delimitador
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        dialect = csv.Sniffer().sniff(sample)

    # Leer archivo con pandas
    df = pd.read_csv(ruta_archivo, sep=dialect.delimiter, dtype=str)

    # Si quedó todo en una sola columna, dividir manualmente
    if df.shape[1] == 1:
        # Tomar los nombres de columnas de la primera fila
        columnas = df.columns[0].split(dialect.delimiter)
        # Dividir cada fila en columnas
        df = df[df.columns[0]].str.split(dialect.delimiter, expand=True)
        df.columns = columnas

    return df

def cargar_csv(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo CSV detectando automáticamente el delimitador.
    Si el DataFrame resultante tiene una sola columna, divide manualmente
    usando el delimitador detectado.
    
    Parámetros:
    -----------
    ruta_archivo : str
        Ruta del archivo CSV.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con los datos del archivo.
    """
    # Detectar delimitador
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        dialect = csv.Sniffer().sniff(sample)

    # Leer archivo con pandas
    df = pd.read_csv(ruta_archivo, sep=dialect.delimiter)

    # Si quedó todo en una sola columna, dividir manualmente
    if df.shape[1] == 1:
        # Tomar los nombres de columnas de la primera fila
        columnas = df.columns[0].split(dialect.delimiter)
        # Dividir cada fila en columnas
        df = df[df.columns[0]].str.split(dialect.delimiter, expand=True)
        df.columns = columnas

    return df