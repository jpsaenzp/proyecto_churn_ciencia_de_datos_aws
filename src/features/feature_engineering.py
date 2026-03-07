import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def crear_variables(df):
    df = df.copy()
    today = pd.Timestamp.today()
    # antigüedad del cliente
    df["customer_tenure_days"] = (today - df["signup_date"]).dt.days
    # recencia de compra
    df["days_since_last_purchase"] = (today - df["last_purchase_date"]).dt.days
    # frecuencia aproximada
    df["shipments_per_month"] = df["total_shipments"] / (df["customer_tenure_days"] / 30 + 1)
    # intensidad de gasto
    df["spend_per_shipment"] = df["monthly_spend"] / (df["total_shipments"] + 1)
    return df


def grafico_matriz_correlacion(df, cols):
    """
    Genera un heatmap de correlación para las columnas especificadas.

    Parámetros:
    df : pandas.DataFrame
        DataFrame de entrada.
    cols : list
        Lista de nombres de columnas a incluir en la matriz de correlación.
    filename : str
        Ruta y nombre del archivo donde se guardará la imagen.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.savefig(f"../imagenes/Matriz de correlación 2.png", bbox_inches="tight")
    plt.show()