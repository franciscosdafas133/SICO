import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Factores de Adicción",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🧪 Analizador de correlación de probar drogas con posibles efectos de este (Trabajo de sicología 2025-1)")

# Función para cargar datos de ejemplo desde el repositorio
@st.cache_data
def load_example_data():
    try:
        url_test = "https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_test.csv"
        url_train = "https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_train.csv"
        test = pd.read_csv(url_test)
        train = pd.read_csv(url_train)
        return test, train
    except Exception as e:
        st.error(f"❌ Error al cargar datos de ejemplo: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Función para procesar los datos
def procesar_datos(test, train):
    imputer = SimpleImputer(strategy='mean')
    columnas_numericas_test = test.select_dtypes(include=np.number).columns
    columnas_numericas_train = train.select_dtypes(include=np.number).columns

    test[columnas_numericas_test] = imputer.fit_transform(test[columnas_numericas_test])
    train[columnas_numericas_train] = imputer.fit_transform(train[columnas_numericas_train])

    # Devuelve datos combinados si se requiere
    return pd.concat([train, test], ignore_index=True)

# Sidebar para subir archivos (opcional)
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("**Carga tus archivos CSV (opcional):**")
    uploaded_test = st.file_uploader("Datos de Prueba", type="csv")
    uploaded_train = st.file_uploader("Datos de Entrenamiento", type="csv")

# Cargar datos del usuario o de ejemplo
if uploaded_test and uploaded_train:
    datos_test = pd.read_csv(uploaded_test)
    datos_train = pd.read_csv(uploaded_train)
    st.success("✅ Usando archivos subidos por el usuario")
else:
    datos_test, datos_train = load_example_data()
    st.warning("⚠️ Usando datos de ejemplo. Sube tus archivos para personalizar el análisis.")

# Procesar datos
datos_procesados = procesar_datos(datos_test, datos_train)

# Validación de columnas esperadas (ajusta según tu caso)
columnas_esperadas = ["Edad", "Consumo_Alcohol", "Consumo_Drogas", "Efectos_Secundarios"]  # ejemplo

if not all(col in datos_procesados.columns for col in columnas_esperadas):
    missing = set(columnas_esperadas) - set(datos_procesados.columns)
    st.error(f"❌ Faltan columnas críticas: {', '.join(missing)}")
else:
    st.success("✅ Datos procesados correctamente")
    st.dataframe(datos_procesados.head())
