# Predicción de Churn de Clientes – Proyecto de Ciencia de Datos End-to-End

Antes que nada es importante que también revise el documento PDF **análisis_y_predicción_de_Churn** ya que contiene más detalle sobre el proyecto.

## 1. Descripción General del Proyecto

Este proyecto desarrolla un **modelo de predicción de abandono de clientes (Customer Churn)** utilizando un pipeline completo de **Machine Learning** y buenas prácticas de **ingeniería de datos y MLOps**.

El objetivo es identificar clientes con **alta probabilidad de abandono** a partir de su comportamiento histórico (gasto, frecuencia de envíos, recencia de compra, entre otros), permitiendo apoyar estrategias de **retención de clientes basadas en datos**.

El flujo de trabajo cubre todo el ciclo de vida de un proyecto de ciencia de datos:

- Limpieza y preparación de datos  
- Análisis exploratorio de datos (EDA)  
- Ingeniería de características (feature engineering)
- Entrenamiento y optimización de hiperparámetros
- Comparación de modelos
- Evaluación de desempeño  
- Persistencia del mejor modelo  
- Funciones de inferencia para nuevos datos  
- Diseño de arquitectura escalable en AWS  

---

# 2. Planteamiento del Problema

La pérdida de clientes representa un impacto directo en los ingresos de una empresa. Identificar de forma temprana clientes con **alto riesgo de abandono** permite implementar **estrategias de retención** y optimizar recursos comerciales.

Este proyecto busca construir un modelo que **estime la probabilidad de churn de cada cliente**, permitiendo priorizar acciones de retención.

---

# 3. Descripción del Conjunto de Datos

El dataset contiene información de clientes con variables relacionadas con su comportamiento dentro del servicio logístico.

| Variable | Descripción |
|--------|-------------|
| customer_id | Identificador único del cliente |
| signup_date | Fecha en que el cliente se registró |
| last_purchase_date | Fecha de última compra |
| monthly_spend | Gasto mensual |
| total_shipments | Número total de envíos |
| churn_label | Variable objetivo (0 = activo, 1 = churn) |

Variables adicionales como **nombre, correo, teléfono y dirección** se utilizaron únicamente para el proceso de limpieza y fueron **excluidas del modelado para proteger la privacidad**.

---

# 4. Limpieza y Preprocesamiento de Datos

El dataset original presentaba varias inconsistencias típicas en datos operacionales.

Las principales etapas de limpieza fueron:

## Estandarización de valores nulos

El dataset contenía múltiples representaciones de valores faltantes:

```
null
Null
NA
n/a
" "
--
```

Todas estas representaciones fueron convertidas a `NaN` para permitir un manejo consistente de valores faltantes.

---

## Normalización de formatos de fecha

Las fechas aparecían en distintos formatos:

```
YYYY-MM-DD
DD/MM/YYYY
MM-DD-YYYY
```

Se implementó una función personalizada de parsing para:

- Detectar automáticamente el año
- Identificar día y mes según restricciones numéricas
- Convertir todas las fechas a un formato estándar

```
YYYY-MM-DD
```

Esto garantiza **consistencia temporal para el análisis**.

---

## Eliminación de duplicados

Se identificaron registros duplicados para algunos clientes.

El procedimiento aplicado fue:

1. Eliminar duplicados exactos.
2. Para registros duplicados por `customer_id`, conservar la fila con **mayor cantidad de valores válidos**.

Esto permite preservar **la información más completa disponible**.

---

## Tratamiento de valores faltantes

Se aplicaron los siguientes supuestos:

| Variable | Tratamiento |
|--------|-------------|
| total_shipments | NaN → 0 (sin envíos) |
| monthly_spend | NaN → 0 (sin compras) |
| churn_label | Eliminado (variable objetivo no puede ser nula) |

---

## Anonimización de variables sensibles

Las siguientes variables contienen **información personal**:

- full_name  
- email  
- phone  
- home_address  

Estas variables **no fueron utilizadas en el modelado** para garantizar privacidad y evitar sesgos.

---

## Estandarización Numérica

Las variables numéricas contenían distintos separadores decimales:

```
255.50
5425,24
300
```

Estos valores fueron normalizados a un formato decimal estándar y convertidos a tipo numérico (float).

---

## Manejo de valores inválidos y atípicos

Durante la limpieza de datos se aplicaron controles básicos para garantizar la consistencia de las variables numéricas.

Primero, se corrigieron valores negativos en `monthly_spend`, ya que el gasto mensual no puede ser menor que cero:

```python
archivo["monthly_spend"] = archivo["monthly_spend"].clip(lower=0)
```

Adicionalmente, se controlaron valores atípicos en `monthly_spend` y `total_shipments` utilizando **winsorización basada en el rango intercuartílico (IQR)**. Este método limita los valores extremos a un rango definido por los cuartiles de la distribución, reduciendo el impacto de outliers sin eliminar observaciones.

---

# 5. Análisis Exploratorio de Datos

Se realizó un análisis exploratorio de datos (EDA) para comprender la estructura y distribución del dataset.

El análisis incluyó:

* distribución de variables numéricas
* análisis de desbalance de clases
* análisis de correlación
* análisis multivariado
* detección de valores atípicos

La distribución de la variable objetivo mostró un desbalance moderado:

```
0 → 75.2%
1 → 24.8%
```

Este desbalance fue considerado al seleccionar las métricas de evaluación.

---

# 6. Ingeniería de Características

Se crearon variables derivadas para capturar mejor el comportamiento del cliente.

### Customer Tenure

Tiempo desde el registro del cliente.

```
customer_tenure_days = today - signup_date
```

### Recency

Tiempo desde la última compra.

```
days_since_last_purchase
```

Esta es **una de las variables más importantes en modelos de churn**.

### Shipment Frequency

Frecuencia de uso del servicio.

```
shipments_per_month
```

### Spend Intensity

Relación entre gasto y número de envíos.

```
spend_per_shipment
```

Estas características ayudan a capturar señales de comportamiento asociadas al churn.

---

# 7. Modelamiento

Se evaluaron múltiples algoritmos de machine learning para identificar el mejor modelo predictivo.

Modelos evaluados:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Support Vector Machine  
- Neural Network (MLP)  
- Deep Neural Network (TensorFlow)

Los hiperparámetros fueron optimizados mediante:

```
GridSearchCV
```

y para la red neuronal:

```
Optuna
```

---

# 8. Preparación de datos para modelamiento

## Train/Test Split

- 70% entrenamiento  
- 30% prueba  

Utilizando **Stratified Split** para preservar la distribución de churn.

---

## Feature Scaling

Se utilizó:

```
StandardScaler
```

para normalizar variables numéricas.

---

## Class Imbalance Handling

Se aplicó:

```
SMOTE
```

para balancear la clase minoritaria (churn).

---

# 9. Métricas de Evaluación

Se utilizaron múltiples métricas:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

El desempeño del modelo también se evaluó mediante:

- ROC Curves  
- Confusion Matrices  

---

# 10. Selección del mejor modelo

Todos los modelos fueron evaluados utilizando el mismo conjunto de prueba.

El mejor modelo se selecciona automáticamente utilizando la métrica:

```
ROC-AUC
```

El modelo ganador se guarda automáticamente.

---

# 11. Persistencia del modelo

El proyecto incluye una función para guardar el mejor modelo:

```
save_best_model()
```

Dependiendo del tipo de modelo se guarda como:

- `.joblib`
- `.keras`

---

# 12. Pipeline de predicción

Se implementó una función que permite:

- Cargar automáticamente el modelo
- Realizar predicciones sobre nuevos datos
- Devolver probabilidad de churn

```
predict_churn()
```

Esto permite utilizar el modelo fácilmente en producción.

---

# 13. Arquitectura en AWS

La solución puede escalarse en AWS mediante:

### Almacenamiento de datos
Amazon S3 como data lake.

### Procesamiento de datos
AWS Glue para procesos ETL.

### Gestión de metadatos
Glue Data Catalog.

### Entrenamiento de modelos
Amazon SageMaker.

### Monitoreo de modelos
Amazon CloudWatch.

### Despliegue del modelo
SageMaker Endpoints para inferencia en tiempo real.

### Analítica
Amazon Athena / Redshift.

### Visualización
Amazon QuickSight o Power BI.

---

<img src="Arquitectura ciencia de datos AWS.png" style="max-width:100%; height:auto;">

---

# 14. Estructura del Proyecto

```
proyecto_churn_ciencia_de_datos_aws/
│
├── databases
│   ├── raw
│   ├── processed
│   └── predictions
│
├── notebooks
│   ├── 01_limpieza_arreglo_datos.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src
│   ├── data
│   ├── features
│   ├── models
│   └── evaluation
│
├── churn_model
│   └── best_model
│
├── imagenes
├── análisis_y_predicción_de_Churn.pdf
├── environment.yml
├── requirements.txt
└── README.md
```

---

# 15. Cómo ejecutar el proyecto

## Clonar el repositorio

```bash
git clone https://github.com/jpsaenzp/proyecto_churn_ciencia_de_datos_aws.git
cd proyecto_churn_ciencia_de_datos_aws
```
## Opción 1: Desde Anaconda

Crear el entorno

```bash
conda env create -f environment.yml
```

Activar el entorno

```bash
conda activate churn-env-data-science
```

## Opción 2: Desde Python, se usa Python 3.11

Crear el entorno

```bash
python -m venv churn-env-data-science
```

Activar el entorno

Mac / Linux
```bash
source churn-env-data-science/bin/activate
```
Windows
```bash
churn-env-data-science\Scripts\activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar Notebooks o Scripts

Flujo recomendado:

```
1️⃣ Limpieza y arreglo de datos
2️⃣ EDA
3️⃣ Feature Engineering
4️⃣ Model Training
5️⃣ Model Evaluation
```

---

# 16. Mejoras Futuras

Existen varias oportunidades para mejorar el modelo:

- Incorporar más variables de comportamiento del cliente  
- Implementar **feature store**  
- Añadir **monitoring de drift**  
- Automatizar el pipeline con **CI/CD**  
- Implementar **retraining automático**  
- Deploy del modelo con **SageMaker endpoints**

---

# 17. Autor

Juan Pablo Saenz Perilla
[jsaenzp@unal.edu.co](mailto:jsaenzp@unal.edu.co)
[LinkedIn](https://www.linkedin.com/in/juan-pablo-saenz-perilla/)