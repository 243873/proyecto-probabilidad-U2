import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from stats_engine import StatsEngine
from visualizer import Visualizer
from ml_model import BayesClassifier

# Inicializar los módulos de tu proyecto
dp = DataProcessor()
stats = StatsEngine()
viz = Visualizer()
ml = BayesClassifier()

# Configuración básica de la página
st.set_page_config(page_title="Analizador Bayesiano", layout="wide")
st.title("Analizador Estadístico y Clasificador Bayesiano")

# 1. Carga de datos
uploaded_file = st.file_uploader("1. Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    # Procesar el archivo subido
    df = dp.load_data(uploaded_file)
    cols_info = dp.detect_columns(df)
    
    st.write("### Explorador de Datos")
    # Mostrar la tabla completa con scroll
    st.dataframe(df, height=250)
    
    # Mostrar tipos de columnas detectados en la barra lateral
    st.sidebar.header("Información de Columnas")
    st.sidebar.write("**Numéricas:**", len(cols_info["numeric"]))
    st.sidebar.write("**Categóricas:**", len(cols_info["categorical"]))

    # Unimos binarias y categóricas porque la encuesta maneja respuestas de texto
    posibles_objetivos = cols_info["binary"] + cols_info["categorical"]
    
    if posibles_objetivos:
        st.markdown("---")
        st.subheader("2. Análisis Probabilístico (Teorema de Bayes)")
        
        col1, col2 = st.columns(2)
        
        # Lado Izquierdo: Configuración del Evento A (Objetivo)
        with col1:
            target = st.selectbox("1. Variable Objetivo (Columna del Evento A)", posibles_objetivos)
            # Extraer las respuestas únicas de esa columna
            valores_unicos_target = df[target].dropna().unique()
            target_value = st.selectbox("2. Valor exacto que representa el Evento A", valores_unicos_target)
            
        # Lado Derecho: Configuración del Evento B (Evidencia)
        with col2:
            feature = st.selectbox("3. Variable de Evidencia (Columna B)", cols_info["numeric"] + cols_info["categorical"])
            
            # Si la evidencia es texto, mostrar selectbox. Si es número, mostrar campo de texto.
            if feature in cols_info["categorical"]:
                valores_unicos_feature = df[feature].dropna().unique()
                threshold = st.selectbox("4. Valor exacto de la Evidencia", valores_unicos_feature)
            else:
                threshold = st.text_input("4. Umbral numérico de Evidencia (Ej. mayor a...)", value="0")

        # Botón de ejecución para Probabilidades
        if st.button("Calcular Probabilidades"):
            p_a, p_b_given_a, p_a_given_b = stats.calculate_bayes(df, target, target_value, feature, threshold)
            
            st.success("Resultados del Teorema de Bayes")
            c1, c2, c3 = st.columns(3)
            c1.metric("P(A) - Probabilidad Base", f"{p_a:.2%}")
            c2.metric("P(B|A) - Prob. Condicional", f"{p_b_given_a:.2%}")
            c3.metric("P(A|B) - Probabilidad Bayesiana", f"{p_a_given_b:.2%}")
            
            st.info(f"**Lectura del resultado:** \n\nLa probabilidad general de que '{target}' sea **'{target_value}'** es del **{p_a:.2%}**.\n\nSin embargo, si sabemos como hecho que '{feature}' es **'{threshold}'**, la probabilidad cambia a **{p_a_given_b:.2%}**.")
            
            # Gráfica de distribución
            st.markdown("---")
            st.subheader("Distribución de los Datos")
            try:
                st.plotly_chart(viz.plot_distribution(df, feature, target), use_container_width=True)
            except Exception as e:
                st.warning("Nota: La gráfica de distribución funciona mejor cuando la variable de evidencia (B) es numérica.")

        # =========================================================================
        # SECCIÓN 3: CLASIFICADOR NAIVE BAYES (Machine Learning)
        # =========================================================================
        st.markdown("---")
        st.subheader("3. Clasificador Automático (Machine Learning - Naive Bayes)")
        st.write("Selecciona múltiples variables para que el modelo aprenda a predecir el evento objetivo.")
        
        # Juntar todas las columnas que pueden servir como predictores
        todas_las_columnas = cols_info["numeric"] + cols_info["categorical"]
        # Remover la columna objetivo para que el modelo no haga "trampa"
        if target in todas_las_columnas:
            todas_las_columnas.remove(target) 
        
        features_selected = st.multiselect("Selecciona variables predictoras (Evidencias)", todas_las_columnas)
        
        if st.button("Entrenar Modelo Bayesiano"):
            if not features_selected:
                st.error("Por favor, selecciona al menos una variable predictora antes de entrenar.")
            else:
                try:
                    # Llamar al modelo
                    cm, acc, sens, esp = ml.evaluate_model(df, target, target_value, features_selected)
                    
                    st.success("¡Modelo entrenado y evaluado con el 30% de los datos!")
                    
                    # Mostrar métricas solicitadas
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Exactitud (Accuracy)", f"{acc:.2%}")
                    col_m2.metric("Sensibilidad (Recall)", f"{sens:.2%}")
                    col_m3.metric("Especificidad", f"{esp:.2%}")
                    
                    with st.expander("¿Qué significan estas métricas?"):
                        st.write(f"- **Accuracy:** El modelo acertó en el **{acc:.2%}** de los casos al predecir el resultado.")
                        st.write(f"- **Sensibilidad:** De todas las veces que *realmente* ocurrió '{target_value}', el modelo lo detectó correctamente el **{sens:.2%}** de las veces.")
                        st.write(f"- **Especificidad:** De todas las veces que *NO* ocurrió '{target_value}', el modelo lo descartó correctamente el **{esp:.2%}** de las veces.")
                    
                    # Mostrar Matriz de Confusión
                    st.markdown("#### Matriz de Confusión")
                    st.plotly_chart(viz.plot_confusion_matrix(cm), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ocurrió un error al entrenar el modelo: {e}")

    else:
        st.warning("No se detectaron columnas válidas para analizar en el CSV.")