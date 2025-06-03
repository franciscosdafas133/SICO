import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import streamlit as st

# Configuración de la página (sin cambios)
st.set_page_config(
    page_title="Analizador de Factores de Adicción",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal (sin cambios)
st.title("🧪 Analizador de correlación de probar drogas con posibles efectos de este(Trabajo de sicología 2025-1)")

# --- Nuevo: Cargar archivos de ejemplo por defecto ---
@st.cache_data
def load_example_data():
    """Carga datos de ejemplo desde el repositorio (rutas relativas)"""
    try:
        # Ejemplo: Cargar desde URLs raw de GitHub (recomendado para despliegue)
        url_test = "https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_test.csv"
        url_train ="https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_train.csv"
        test = pd.read_csv(url_test)
        train = pd.read_csv(url_train)
        return test, train
    except:
        return test, train

# Sidebar para controles (sin cambios, excepto mensaje añadido)
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("**Carga tus archivos CSV (opcional):**")  # Cambiado a "opcional"
    uploaded_test = st.file_uploader("Datos de Prueba", type="csv")
    uploaded_train = st.file_uploader("Datos de Entrenamiento", type="csv")
    
    # Resto del sidebar (sin cambios)...

# --- Modificación en el bloque principal ---
if uploaded_test and uploaded_train:
    # Usar archivos subidos por el usuario
    datos_test = pd.read_csv(uploaded_test)
    datos_train = pd.read_csv(uploaded_train)
    st.success("✅ Usando archivos subidos por el usuario")
else:
    # Cargar datos de ejemplo si no hay archivos subidos
    datos_test, datos_train = load_example_data()
    st.warning("⚠️ Usando datos de ejemplo. Sube tus archivos para personalizar el análisis.")

# --- Resto del código (procesamiento, pestañas, gráficos) permanece IGUAL ---
datos_procesados = procesar_datos(datos_test, datos_train)

# Verificación de columnas y todo lo demás (sin cambios)...
if not all(col in datos_procesados.columns for col in columnas_esperadas):
    missing = set(columnas_esperadas) - set(datos_procesados.columns)
    st.error(f"Faltan columnas críticas: {', '.join(missing)}")
else:
    st.success("✅ Datos procesados correctamente")
    
    # Pestañas y gráficos (sin cambios)...
