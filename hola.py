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

# URLs de los archivos en GitHub (actualiza con tus URLs reales)
TEST_DATA_URL = "https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_test.csv"
TRAIN_DATA_URL = "https://raw.githubusercontent.com/franciscosdafas133/SICO/main/student_addiction_dataset_train.csv"

@st.cache_data
def load_data():
    """Carga los datos desde GitHub"""
    try:
        test_data = pd.read_csv(TEST_DATA_URL)
        train_data = pd.read_csv(TRAIN_DATA_URL)
        return test_data, train_data
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("**Datos cargados directamente desde GitHub**")
    
    st.markdown("---")
    st.markdown("**Opciones de análisis:**")
    variable_principal = st.selectbox(
        "Variable principal para análisis:",
        options=[
            'Experimentation', 'Academic_Performance_Decline', 'Social_Isolation',
            'Financial_Issues', 'Physical_Mental_Health_Problems',
            'Legal_Consequences', 'Relationship_Strain', 'Risk_Taking_Behavior',
            'Withdrawal_Symptoms', 'Denial_and_Resistance_to_Treatment',
            'Addiction_Class'
        ],
        index=0
    )
    
    variables_correlacion = st.multiselect(
        "Selecciona variables para ver su correlación con Experimentation:",
        options=[
            'Academic_Performance_Decline', 'Social_Isolation',
            'Financial_Issues', 'Physical_Mental_Health_Problems',
            'Legal_Consequences', 'Relationship_Strain', 'Risk_Taking_Behavior',
            'Withdrawal_Symptoms', 'Denial_and_Resistance_to_Treatment',
            'Addiction_Class'
        ],
        default=['Academic_Performance_Decline', 'Risk_Taking_Behavior', 'Addiction_Class']
    )
    
    umbral_correlacion = st.slider("Umbral de correlación mínima:", 0.0, 1.0, 0.2, 0.01)

# Procesamiento de datos (igual que antes)
def procesar_datos(datos_test, datos_train):
    datos = pd.concat([datos_train, datos_test], axis=0)
    datos = datos.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
    conversion = {'sí': 1, 'si': 1, 'yes': 1, 'no': 0, 'not': 0}
    datos = datos.replace(conversion).apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='most_frequent')
    datos_imputados = imputer.fit_transform(datos)
    return pd.DataFrame(datos_imputados, columns=datos.columns)

# Cargar y procesar datos
datos_test, datos_train = load_data()

if datos_test is not None and datos_train is not None:
    datos_procesados = procesar_datos(datos_test, datos_train)
    
    columnas_esperadas = [
        'Experimentation', 'Academic_Performance_Decline', 'Social_Isolation',
        'Financial_Issues', 'Physical_Mental_Health_Problems',
        'Legal_Consequences', 'Relationship_Strain', 'Risk_Taking_Behavior',
        'Withdrawal_Symptoms', 'Denial_and_Resistance_to_Treatment',
        'Addiction_Class'
    ]
    
    if not all(col in datos_procesados.columns for col in columnas_esperadas):
        missing = set(columnas_esperadas) - set(datos_procesados.columns)
        st.error(f"Faltan columnas críticas: {', '.join(missing)}")
    else:
        st.success("✅ Datos cargados y procesados correctamente desde GitHub")
        
        # Resto del código (pestañas, gráficos, etc.) permanece igual
        tab1, tab2, tab3 = st.tabs(["🔍 Correlaciones", "📊 Distribución", "📝 Reporte"])
        
        with tab1:
            st.header(f"Correlaciones con {variable_principal}")
            corr = datos_procesados.corr()
            
            if variable_principal == 'Experimentation' and variables_correlacion:
                variables_analisis = ['Experimentation'] + variables_correlacion
                corr_filtrada = datos_procesados[variables_analisis].corr()
                correlaciones = corr_filtrada['Experimentation'].sort_values(ascending=False)
            else:
                correlaciones = corr[variable_principal].sort_values(ascending=False)
            
            correlaciones_filtradas = correlaciones[abs(correlaciones) >= umbral_correlacion]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Correlación más alta",
                    value=f"{correlaciones[1]:.2f}",
                    delta=f"con {correlaciones.index[1]}"
                )
            with col2:
                st.metric(
                    label="Correlación más baja",
                    value=f"{correlaciones[-1]:.2f}",
                    delta=f"con {correlaciones.index[-1]}"
                )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=correlaciones_filtradas.values,
                y=correlaciones_filtradas.index,
                palette="vlag",
                ax=ax
            )
            title = f"Correlaciones significativas (≥{umbral_correlacion})"
            if variable_principal == 'Experimentation' and variables_correlacion:
                title += " - Variables seleccionadas"
            ax.set_title(title)
            ax.set_xlim(-1, 1)
            ax.axvline(0, color="black", linestyle="--")
            st.pyplot(fig)
            
            st.subheader("Matriz de Correlación")
            plt.figure(figsize=(12, 8))
            if variable_principal == 'Experimentation' and variables_correlacion:
                sns.heatmap(
                    corr_filtrada,
                    annot=True,
                    cmap="vlag",
                    fmt=".2f",
                    linewidths=0.5,
                    vmin=-1,
                    vmax=1,
                    center=0
                )
            else:
                sns.heatmap(
                    corr,
                    annot=True,
                    cmap="vlag",
                    fmt=".2f",
                    linewidths=0.5,
                    vmin=-1,
                    vmax=1,
                    center=0
                )
            st.pyplot(plt)
        
        with tab2:
            st.header("Distribución de Variables")
            variable_dist = st.selectbox(
                "Seleccionar variable para distribución:",
                options=columnas_esperadas,
                index=0
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                x=datos_procesados[variable_dist],
                ax=ax,
                palette="viridis"
            )
            ax.set_title(f"Distribución de {variable_dist}")
            st.pyplot(fig)
            
            st.subheader("Estadísticas")
            st.dataframe(
                datos_procesados[columnas_esperadas].describe(),
                use_container_width=True
            )
        
        with tab3:
            st.header("Reporte de Análisis")
            st.subheader("Hallazgos Principales")
            top_positiva = corr[variable_principal].nlargest(2).iloc[1]
            top_negativa = corr[variable_principal].nsmallest(1).iloc[0]
            st.markdown(f"""
            - **Correlación más fuerte positiva**: {top_positiva.name} ({top_positiva:.2f})
            - **Correlación más fuerte negativa**: {top_negativa.name} ({top_negativa:.2f})
            """)
            
            st.subheader(f"Análisis para {variable_principal}")
            dist = datos_procesados[variable_principal].value_counts(normalize=True)
            st.markdown(f"""
            - **Distribución**: 
              - Sí: {dist.get(1, 0)*100:.1f}%
              - No: {dist.get(0, 0)*100:.1f}%
            """)

# Pie de página
st.divider()
st.caption("""
Herramienta desarrollada para el análisis de factores de adicción estudiantil. 
Los datos se han convertido a valores binarios (1 = Sí, 0 = No) para el análisis.
Datos cargados automáticamente desde GitHub.
""")
