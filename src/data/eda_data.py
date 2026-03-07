import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr


def perfil_dataset(df):
    profile = pd.DataFrame({"dtype": df.dtypes, "n_unique": df.nunique(),
                            "n_nulls": df.isnull().sum(),
                            "pct_nulls": df.isnull().mean() * 100})
    profile["sample_values"] = df.apply(lambda x: x.dropna().unique()[:3])
    return profile.sort_values("pct_nulls", ascending=False)


def eda_datetime(df, datetime_cols):
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        trend = df.groupby(df[col].dt.to_period("M")).size()
        plt.figure()
        trend.plot()
        plt.title(f"Tendencia mensual - {col}")
        plt.savefig(f"../imagenes/Tendencia mensual - {col}.png", bbox_inches="tight")
        plt.show()


def crear_variables_temporales(df):
    df = df.copy()
    today = pd.Timestamp.today()
    df["customer_tenure_days"] = (today - df["signup_date"]).dt.days
    df["days_since_last_purchase"] = (today - df["last_purchase_date"]).dt.days
    df["signup_month"] = df["signup_date"].dt.to_period("M").dt.to_timestamp()
    df["purchase_month"] = df["last_purchase_date"].dt.to_period("M").dt.to_timestamp()
    return df


def eda_numerico(df, numeric_cols):
    summary = df[numeric_cols].describe().T
    summary["skew"] = df[numeric_cols].skew()
    summary["kurtosis"] = df[numeric_cols].kurt()
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribución - {col}")
        plt.savefig(f"../imagenes/Distribución - {col}.png", bbox_inches="tight")
        plt.show()
    return summary


def eda_categorico(df, categorical_cols):
    results = {}
    for col in categorical_cols:
        freq = df[col].value_counts(dropna=False)
        results[col] = freq
        plt.figure()
        freq.head(20).plot(kind="bar")
        plt.title(f"Top Categorías - {col}")
        plt.savefig(f"../imagenes/Top Categorías - {col}.png", bbox_inches="tight")
        plt.show()
    return results


def churn_resumen(df):
    features = ["monthly_spend", "total_shipments", "customer_tenure_days", "days_since_last_purchase"]
    
    lista_resumenes = []
    for feature in features:
        resumen = df.groupby("churn_label")[feature].describe().reset_index()
        # Agregar columna con el nombre de la feature
        resumen["feature"] = feature
        lista_resumenes.append(resumen)
    
    # Concatenar todos los DataFrames
    df_final = pd.concat(lista_resumenes, ignore_index=True)
    
    return df_final


def churn_analisis_temporal(df):
    df = df.copy()
    df["signup_month"] = df["signup_date"].dt.to_period("M")
    churn_rate = (df.groupby("signup_month")["churn_label"].mean().reset_index())
    churn_rate["signup_month"] = churn_rate["signup_month"].astype(str)
    return churn_rate


def grafico_gasto_vs_churn(df):
    sns.boxplot(x="churn_label", y="monthly_spend", data=df)
    plt.title("Gasto mensual vs churn")
    plt.savefig(f"../imagenes/Gasto mensual vs churn.png", bbox_inches="tight")
    plt.show()


def grafico_shipments_vs_churn(df):
    sns.boxplot(x="churn_label", y="total_shipments", data=df)
    plt.title("Total Shipments vs Churn")
    plt.savefig(f"../imagenes/Total Shipments vs Churn.png", bbox_inches="tight")
    plt.show()


def preparar_series_de_tiempo(df):
    df = df.copy()
    # crear periodo mensual
    df["month"] = df["last_purchase_date"].dt.to_period("M").dt.to_timestamp()
    # agregación por mes y churn
    ts = (df.groupby(["month", "churn_label"]).agg({"monthly_spend": "mean", "total_shipments": "mean"}).reset_index())
    return ts


def grafico_series_tiempo_multivariada(df):
    ts = (df.groupby(["purchase_month","churn_label"]).agg({"monthly_spend":"mean","total_shipments":"mean"}).reset_index())
    fig, axes = plt.subplots(2,1, figsize=(10,8))
    # gasto mensual
    sns.lineplot(data=ts, x="purchase_month", y="monthly_spend", hue="churn_label", marker="o", ax=axes[0])
    axes[0].set_title("Gasto mensual en el tiempo por Churn")
    axes[0].set_ylabel("Gasto mensual")
    # envíos
    sns.lineplot(data=ts, x="purchase_month", y="total_shipments", hue="churn_label", marker="o", ax=axes[1])
    axes[1].set_title("Envíos totales en el tiempo por Churn")
    axes[1].set_ylabel("Envíos totales")
    plt.tight_layout()
    plt.savefig(f"../imagenes/Gasto envio mensual por Churn.png", bbox_inches="tight")
    plt.show()


def graficar_outliers(df):
    fig, ax = plt.subplots(2,2, figsize=(13,8))
    sns.boxplot(y=df["monthly_spend"], ax=ax[0,0])
    ax[0,0].set_title("Monthly Spend Outliers")
    sns.boxplot(y=df["total_shipments"], ax=ax[0,1])
    ax[0,1].set_title("Total Shipments Outliers")
    sns.boxplot(y=df["customer_tenure_days"], ax=ax[1,0])
    ax[1,0].set_title("Días de permanencia del cliente")
    sns.boxplot(y=df["days_since_last_purchase"], ax=ax[1,1])
    ax[1,1].set_title("Días desde la última compra")
    plt.savefig(f"../imagenes/análisis outliers.png", bbox_inches="tight")
    plt.show()


def matriz_correlacion(df):
    cols = ["monthly_spend", "total_shipments", "customer_tenure_days", "days_since_last_purchase", "churn_label"]
    corr = df[cols].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.savefig(f"../imagenes/Matriz de correlación.png", bbox_inches="tight")
    plt.show()


def churn_correlacion_numerica(df):
    variables = ["monthly_spend", "total_shipments", "customer_tenure_days", "days_since_last_purchase"]
    results = {}
    for v in variables:
        # Filtrar filas donde la columna v no sea NaN
        mask = df[v].notna()
        x = df.loc[mask, "churn_label"]
        y = df.loc[mask, v]
        # Calcular correlación
        corr, pval = pointbiserialr(x, y)
        results[v] = {"correlation": corr, "p_value": pval, "n": mask.sum()}
    return pd.DataFrame(results).T


def lifecycle_analysis(df):
    churn = df[df["churn_label"]==1]
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="days_since_last_purchase", y="monthly_spend", size="total_shipments", data=churn)
    plt.title("Ciclo de vida del cliente antes de la pérdida")
    plt.savefig(f"../imagenes/Ciclo de vida del cliente antes de la pérdida.png", bbox_inches="tight")
    plt.show()