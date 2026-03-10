import streamlit as st
import pandas as pd
import time
from data_processor import DataProcessor
from stats_engine import StatsEngine
from visualizer import Visualizer
from ml_model import BayesClassifier

# 1. Configuracion de la pagina
st.set_page_config(page_title="DataBayes Pro", layout="wide")

# 2. CSS Avanzado 
st.markdown("""
    <style>
        [data-testid="stHeader"] { display: none; }
        .main-header {
            position: fixed; top: 0; left: 0; width: 100%;
            background: linear-gradient(90deg, #1f77b4 0%, #00d4ff 100%);
            color: white; z-index: 9999; padding: 15px 0;
            text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .main-header h1 { margin: 0; font-size: 28px; font-weight: 700; }
        .main .block-container { padding-top: 100px !important; }
        .bayes-formula {
            background-color: #f0f7ff; padding: 15px; border-radius: 10px;
            border: 1px dashed #1f77b4; text-align: center; font-size: 20px;
            margin-bottom: 20px;
        }
    </style>
    <div class="main-header"><h1>ANALIZADOR BAYESIANO</h1></div>
""", unsafe_allow_html=True)

# Inicializar componentes
dp = DataProcessor()
stats = StatsEngine()
viz = Visualizer()
ml = BayesClassifier()

# 3. Seccion principal de carga (Siempre visible al inicio)
st.subheader("Carga de Base de Datos")
archivo = st.file_uploader("Arrastra tu archivo CSV aqui o haz clic para buscar", type="csv")

# Si no hay archivo, mostramos un mensaje y no cargamos nada mas
if archivo is None:
    st.info("Por favor, sube un archivo CSV para habilitar las secciones de analisis y visualizacion.")
else:
    # 4. Animacion de carga y lectura de archivo
    with st.spinner('Procesando datos, detectando columnas y limpiando registros...'):
        time.sleep(1.2) # Pausa para visualizar la animacion de carga
        df = dp.load_data(archivo)
        cols = dp.detect_columns(df)
        
    # Mostrar confirmacion con el nombre del archivo
    st.success(f"Archivo '{archivo.name}' cargado exitosamente. Total de registros: {len(df)}")
    
    # 5. Creacion de Ventanas (Solo aparecen cuando el archivo ya esta cargado)
    tab1, tab2, tab3 = st.tabs(["1. Gestion de Datos", "2. Analisis Bayesiano", "3. Machine Learning"])

    # ==========================================
    # VENTANA 1: GESTION DE DATOS
    # ==========================================
    with tab1:
        st.subheader("Exploracion de Base de Datos")
        with st.container(border=True):
            st.dataframe(df, use_container_width=True)

    opciones = cols["binary"] + cols["categorical"]
    
    # ==========================================
    # VENTANA 2: TEOREMA DE BAYES
    # ==========================================
    with tab2:
        st.subheader("Motor de Probabilidades Condicionales")
        
        with st.expander("Que es el Teorema de Bayes? (Haz clic para aprender)"):
            st.markdown("""
            El Teorema de Bayes nos permite actualizar la probabilidad de un evento a medida que aparece nueva evidencia.
            """)
            st.markdown(r'<div class="bayes-formula">P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}</div>', unsafe_allow_html=True)
            st.markdown("""
            * **P(A|B)**: Probabilidad de que ocurra A sabiendo que paso B (lo que queremos descubrir).
            * **P(A)**: Probabilidad inicial del evento (sin saber nada mas).
            * **P(B|A)**: Probabilidad de ver la evidencia si el evento es cierto.
            """)

        if opciones:
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    target = st.selectbox("Evento Principal (A)", opciones)
                    val_a = st.selectbox("Valor deseado para el Evento A", df[target].unique())
                with c2:
                    feat = st.selectbox("Evidencia Observada (B)", cols["numeric"] + cols["categorical"])
                    if feat in cols["categorical"]:
                        val_b = st.selectbox("Valor de la evidencia B", df[feat].unique())
                    else:
                        val_b = st.text_input("Umbral de evidencia B (mayor a...)", value="0")

                if st.button("Calcular Teorema de Bayes", type="primary"):
                    pa, pba, pab = stats.calculate_bayes(df, target, val_a, feat, val_b)
                    pb = (pba * pa) / pab if pab > 0 else 0

                    st.markdown("---")
                    st.subheader("Resultados del Analisis")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probabilidad Base P(A)", f"{pa:.2%}")
                    m2.metric("Condicional P(B|A)", f"{pba:.2%}")
                    m3.metric("Resultado Final P(A|B)", f"{pab:.2%}")
                    
                    st.write("### Desglose del Calculo Paso a Paso")
                    st.markdown(f"""
                    Para llegar a este resultado, el sistema aplico la formula matematica con tus datos:
                    
                    $$P(A|B) = \\frac{{{pba:.4f} \\cdot {pa:.4f}}}{{{pb:.4f}}} = {pab:.4f}$$
                    
                    **Interpretacion Detallada:**
                    1. **La Base:** Sin saber ninguna otra informacion, la probabilidad general de que '{target}' sea '{val_a}' es del **{pa:.2%}**.
                    2. **La Evidencia:** Analizamos la evidencia '{feat}' = '{val_b}'. Nos dimos cuenta de que cuando el evento A si ocurre, esta evidencia esta presente el **{pba:.2%}** de las veces.
                    3. **La Conclusion:** Al integrar esta evidencia, nuestra certeza cambia. Ahora sabemos que si observamos la evidencia '{val_b}', la probabilidad real de que ocurra '{val_a}' se actualiza al **{pab:.2%}**.
                    """)
                    
                    st.plotly_chart(viz.plot_distribution(df, feat, target), use_container_width=True)

    # ==========================================
    # VENTANA 3: MACHINE LEARNING
    # ==========================================
    with tab3:
        st.subheader("Clasificador Inteligente (Naive Bayes)")
        st.write("Selecciona multiples variables para entrenar a la IA a predecir el evento objetivo.")
        
        with st.container(border=True):
            preds = st.multiselect("Variables predictoras (Evidencias)", [c for c in df.columns if c != target])
            
            if st.button("Entrenar Modelo Predictivo", type="primary"):
                if preds:
                    with st.spinner('Entrenando algoritmo clasificador con tus datos...'):
                        time.sleep(1)
                        cm, acc, sen, esp = ml.evaluate_model(df, target, val_a, preds)
                    
                    st.success("Modelo entrenado con exito.")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Exactitud (Accuracy)", f"{acc:.2%}")
                    col_m2.metric("Sensibilidad", f"{sen:.2%}")
                    col_m3.metric("Especificidad", f"{esp:.2%}")
                    
                    st.markdown("### Conclusion del Modelo Predictivo")
                    
                    if acc >= 0.8:
                        nivel = "alta"
                    elif acc >= 0.6:
                        nivel = "moderada"
                    else:
                        nivel = "baja"
                        
                    st.info(f"""
                    Basado en las metricas obtenidas, el modelo de inteligencia artificial ha logrado una capacidad de prediccion **{nivel}** (Exactitud del {acc:.2%}). 
                    
                    * **Lo que hace bien:** Tiene una Especificidad del **{esp:.2%}**, lo que significa que el algoritmo es capaz de descartar correctamente la gran mayoria de casos donde el evento '{val_a}' NO va a ocurrir.
                    * **Areas de mejora:** Su Sensibilidad es del **{sen:.2%}**. Esto indica que porcentaje de los eventos '{val_a}' reales logro detectar. Si este numero es bajo, sugiere que las variables predictoras seleccionadas no son suficientes para que la IA entienda el patron completo.
                    """)
                    
                    st.plotly_chart(viz.plot_confusion_matrix(cm), use_container_width=True)
                else:
                    st.warning("Selecciona al menos una variable predictora.")