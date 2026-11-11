import os
import math
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Dirección de la API (en tu computadora o en internet)
API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Riesgo de enfermedad cardíaca", page_icon="❤️", layout="wide")

# ---------- NOMBRES Y OPCIONES ----------
NOMBRES = {
    "edad": "Edad", "sexo": "Sexo", "tipo_dolor": "Tipo de dolor de pecho",
    "presion": "Presión en reposo", "colesterol": "Colesterol",
    "azucar": "Azúcar en ayunas", "ecg": "Electrocardiograma",
    "pulso_max": "Pulso máximo", "angina": "Angina por ejercicio",
    "depresion_st": "Depresión ST", "pendiente": "Pendiente ST",
    "vasos": "Vasos coloreados", "talasemia": "Talasemia",
}
SEXO = {0: "Mujer", 1: "Hombre"}
SINO = {0: "No", 1: "Sí"}
DOLOR = {1: "Angina típica", 2: "Angina atípica", 3: "Dolor no anginoso", 4: "Sin síntomas"}
ECG = {0: "Normal", 1: "Anomalía ST-T", 2: "Hipertrofia"}
PENDIENTE = {1: "Ascendente", 2: "Plana", 3: "Descendente"}
TALASEMIA = {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"}
TEXTOS = {"sexo": SEXO, "tipo_dolor": DOLOR, "azucar": SINO, "ecg": ECG,
          "angina": SINO, "pendiente": PENDIENTE, "talasemia": TALASEMIA}

# Valores con los que empieza el formulario del paciente
VALORES_INICIALES = {
    "edad": 55, "sexo": 1, "tipo_dolor": 3, "presion": 130, "colesterol": 240,
    "azucar": 0, "ecg": 0, "pulso_max": 150, "angina": 0, "depresion_st": 1.0,
    "pendiente": 2, "vasos": 0, "talasemia": 3,
}
EJEMPLO_A = {
    "edad": 63, "sexo": 1, "tipo_dolor": 1, "presion": 145, "colesterol": 233,
    "azucar": 1, "ecg": 2, "pulso_max": 150, "angina": 0, "depresion_st": 2.3,
    "pendiente": 3, "vasos": 0, "talasemia": 6,
}
EJEMPLO_B = {
    "edad": 67, "sexo": 1, "tipo_dolor": 4, "presion": 160, "colesterol": 286,
    "azucar": 0, "ecg": 2, "pulso_max": 108, "angina": 1, "depresion_st": 1.5,
    "pendiente": 2, "vasos": 3, "talasemia": 3,
}


# ---------- FUNCIONES AUXILIARES ----------
def llamar_api(ruta, datos):
    # Envía datos a la API y devuelve la respuesta (o None si falla)
    try:
        respuesta = requests.post(API + ruta, json=datos, timeout=60)
        respuesta.raise_for_status()
        st.session_state["conectada"] = True
        return respuesta.json()
    except Exception:
        st.session_state["conectada"] = False
        return None


def consultar_metricas():
    # Pregunta a la API si está encendida y si ya tiene métricas guardadas
    try:
        respuesta = requests.get(API + "/metricas", timeout=60)
        respuesta.raise_for_status()
        datos = respuesta.json()
        if "exactitud" in datos:
            return True, datos
        return True, None
    except Exception:
        return False, None


def cargar_ejemplo(valores):
    # Pone unos valores en el formulario del paciente
    for clave, valor in valores.items():
        st.session_state[clave] = valor
    st.session_state["resultado"] = None


def dibujar_medidor(prob):
    # Medidor de semicírculo: verde = riesgo bajo, rojo = riesgo alto
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.add_patch(Wedge((0, 0), 1, 90, 180, width=0.3, color="#2e9e5b"))
    ax.add_patch(Wedge((0, 0), 1, 0, 90, width=0.3, color="#d64545"))
    angulo = math.radians(180 - 180 * prob)
    ax.plot([0, 0.8 * math.cos(angulo)], [0, 0.8 * math.sin(angulo)],
            color="#333333", linewidth=3)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#333333"))
    ax.text(0, -0.25, f"{prob:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold")
    ax.text(-0.85, -0.08, "Bajo", ha="center", fontsize=10)
    ax.text(0.85, -0.08, "Alto", ha="center", fontsize=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.45, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ---------- MEMORIA DE LA PÁGINA ----------
if "metricas" not in st.session_state:
    st.session_state["metricas"] = None
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None
for clave, valor in VALORES_INICIALES.items():
    st.session_state.setdefault(clave, valor)

# Al abrir la página por primera vez, se conecta con la API
if "conectada" not in st.session_state:
    with st.spinner("Conectando con la API..."):
        conectada, metricas_api = consultar_metricas()
    st.session_state["conectada"] = conectada
    st.session_state["metricas"] = metricas_api

# ---------- ENCABEZADO ----------
st.title("Sistema de predicción de riesgo de enfermedad cardíaca")
st.write("Aplicación de Machine Learning con Random Forest y datos clínicos reales.")

# ---------- BARRA LATERAL ----------
st.sidebar.header("⚙️ Configuración del modelo")

n_arboles = st.sidebar.slider(
    "🌳 Número de árboles", 10, 300, 100,
    help="Cuántos 'médicos' votan para decidir. Más árboles suele dar un resultado más estable.")
profundidad = st.sidebar.slider(
    "📏 Profundidad máxima", 1, 15, 5,
    help="Cuántas preguntas seguidas puede hacer cada 'médico' antes de decidir.")

if st.sidebar.button("🚀 Entrenar modelo", type="primary"):
    with st.spinner("Entrenando el modelo..."):
        respuesta = llamar_api("/entrenar", {"n_arboles": n_arboles, "profundidad": profundidad})
    if respuesta is None:
        st.sidebar.error("No se pudo entrenar. Revisa que la API esté encendida.")
    else:
        st.session_state["metricas"] = respuesta
        st.sidebar.success("✅ Modelo entrenado correctamente.")

if st.session_state["conectada"]:
    st.sidebar.write("🟢 Estado de la API: conectada")
else:
    st.sidebar.write("🔴 Estado de la API: desconectada")

# ---------- PESTAÑAS ----------
tab_info, tab_res, tab_pred = st.tabs(
    ["Acerca del proyecto", "Resultados del entrenamiento", "Predicción de un paciente"])

# ===== PESTAÑA 1: ACERCA DEL PROYECTO =====
with tab_info:
    st.subheader("¿Qué es este proyecto?")
    st.write("Es una aplicación web que usa Inteligencia Artificial para estimar si una persona "
             "tiene un riesgo alto o bajo de sufrir una enfermedad del corazón, a partir de 13 "
             "datos clínicos. Fue desarrollada como proyecto final de la asignatura "
             "Proyectos de Machine Learning.")

    st.subheader("¿De qué se trata?")
    st.write("El sistema aprende de casos reales de pacientes: revisa qué datos tenían las "
             "personas sanas y cuáles las personas enfermas, y con eso aprende a reconocer "
             "patrones. Después, puedes ingresar los datos de un paciente nuevo y el sistema "
             "te dice si se parece más a un caso sano o a un caso enfermo, junto con un "
             "porcentaje de probabilidad.")
    st.write("Desde esta aplicación puedes ajustar el modelo, entrenarlo, ver qué tan bien "
             "funciona y probar pacientes nuevos, sin escribir nada de código.")

    st.subheader("¿De dónde vienen los datos?")
    d1, d2, d3 = st.columns(3)
    d1.metric("Pacientes reales", "303")
    d2.metric("Datos por paciente", "13")
    d3.metric("Pacientes sanos / enfermos", "164 / 139")
    st.write("Se usó el conjunto de datos *Heart Disease* (versión Cleveland), publicado por la "
             "Universidad de California, Irvine (UCI). Son datos reales de pacientes atendidos "
             "en hospitales, no inventados ni generados con IA.")
    st.caption("Fuente: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). "
               "Heart Disease [Dataset]. UCI Machine Learning Repository. "
               "https://doi.org/10.24432/C52P4X")
    st.write("Antes de entrenar, los datos se ordenaron: se les puso nombre a las columnas, se "
             "completaron los pocos datos vacíos y se separó el 80 % para que el modelo aprenda "
             "y el 20 % para ponerlo a prueba con casos que nunca vio.")

    st.subheader("¿Cómo está hecho?")
    st.write("El sistema tiene tres piezas que trabajan juntas, como en un restaurante:")
    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown("**1. El modelo (la cocina)**")
            st.write("Es un Random Forest, que funciona como un grupo de médicos: cada uno "
                     "revisa los datos del paciente y al final gana la opinión de la mayoría.")
    with b:
        with st.container(border=True):
            st.markdown("**2. La API (el mesero)**")
            st.write("Recibe los datos desde la pantalla, se los lleva al modelo y regresa con "
                     "la respuesta. Tiene tres funciones: entrenar, mostrar métricas y predecir.")
    with c:
        with st.container(border=True):
            st.markdown("**3. La interfaz (el comedor)**")
            st.write("Es esta pantalla. Permite ajustar el modelo, ver los resultados y "
                     "probar pacientes de forma sencilla.")
    st.caption("Hecho con Python: scikit-learn (modelo), FastAPI (API) y Streamlit (interfaz).")

    st.subheader("Cómo usar la aplicación")
    st.write("1. En la barra lateral, ajusta los controles y presiona **Entrenar modelo**.")
    st.write("2. Entra a **Resultados del entrenamiento** para ver qué tan bien funciona.")
    st.write("3. Entra a **Predicción de un paciente**, ingresa los datos y presiona "
             "**Realizar predicción**.")

    st.subheader("Integrantes")
    st.write("Liseth Estefania Nato Paucar, Andrea Abigail Caisa Cango y "
             "Josselyn Andrea Quiñonez Ushiña.")
    st.caption("Esta aplicación es un ejercicio educativo y no reemplaza el diagnóstico "
               "de un profesional de la salud.")

# ===== PESTAÑA 2: RESULTADOS =====
with tab_res:
    m = st.session_state["metricas"]

    if m is None:
        st.info("Todavía no hay resultados. Presiona '🚀 Entrenar modelo' en la barra lateral.")
    else:
        # --- Métricas ---
        st.subheader("Métricas del modelo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exactitud", f"{m['exactitud']:.1%}",
                  help="De cada 100 pacientes, a cuántos acierta el modelo.")
        c2.metric("Precisión", f"{m['precision']:.1%}",
                  help="Cuando el modelo dice 'enfermo', qué tan seguido tiene razón.")
        c3.metric("Sensibilidad (Recall)", f"{m['recall']:.1%}",
                  help="De los pacientes realmente enfermos, a cuántos detecta.")
        c4.metric("F1", f"{m['f1']:.1%}",
                  help="Un promedio que equilibra la precisión y la sensibilidad.")
        st.caption("Medidas con el 20 % de pacientes que el modelo nunca vio durante el entrenamiento.")

        st.divider()

        # --- Matriz de confusión ---
        st.subheader("Matriz de confusión")
        matriz = m["matriz"]

        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.imshow(matriz, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sano", "Enfermo"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sano", "Enfermo"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Diagnóstico real")
        maximo = max(max(fila) for fila in matriz)
        for i in range(2):
            for j in range(2):
                color = "white" if matriz[i][j] > maximo / 2 else "black"
                ax.text(j, i, matriz[i][j], ha="center", va="center", fontsize=14, color=color)
        fig.tight_layout()

        col_g, col_t = st.columns(2)
        with col_g:
            st.pyplot(fig)
        plt.close(fig)
        with col_t:
            st.markdown("**Cómo leer la matriz**")
            st.write("Cada fila es lo que el paciente realmente tenía, y cada columna es lo "
                     "que dijo el modelo.")
            st.write(f"- **Aciertos (diagonal):** {matriz[0][0]} sanos bien clasificados y "
                     f"{matriz[1][1]} enfermos bien detectados.")
            st.write(f"- **Falsas alarmas (arriba a la derecha):** {matriz[0][1]} personas "
                     "sanas que el modelo marcó como enfermas.")
            st.write(f"- **Enfermos no detectados (abajo a la izquierda):** {matriz[1][0]}. "
                     "Es el error más delicado, porque esa persona podría quedarse sin atención.")
            st.info("Mientras más grandes sean los números de la diagonal y más pequeños los "
                    "otros dos, mejor funciona el modelo.")

        st.divider()

        # --- Dos gráficos de apoyo ---
        st.subheader("Gráficos de apoyo")
        g1, g2 = st.columns(2)

        # Gráfico 1: comparación de las métricas
        with g1:
            st.markdown("**Comparación de las métricas**")
            nombres = ["Exactitud", "Precisión", "Recall", "F1"]
            valores = [m["exactitud"], m["precision"], m["recall"], m["f1"]]
            fig2, ax2 = plt.subplots(figsize=(5, 3.4))
            barras = ax2.bar(nombres, [v * 100 for v in valores],
                             color=["#2f5496", "#2e9e5b", "#e08a1e", "#8e5bd1"])
            ax2.set_ylim(0, 110)
            ax2.set_ylabel("Porcentaje (%)")
            for barra, v in zip(barras, valores):
                ax2.text(barra.get_x() + barra.get_width() / 2, v * 100 + 2,
                         f"{v:.1%}", ha="center", fontsize=9)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            pares = dict(zip(nombres, valores))
            mejor = max(pares, key=pares.get)
            peor = min(pares, key=pares.get)
            st.write(f"La medida más alta es **{mejor}** ({pares[mejor]:.1%}) y la más baja es "
                     f"**{peor}** ({pares[peor]:.1%}). Mientras más cerca del 100 % esté "
                     "cada barra, mejor funciona el modelo.")

        # Gráfico 2: qué datos pesan más
        with g2:
            st.markdown("**Qué datos pesan más en la decisión**")
            importancias = m.get("importancias")
            if importancias:
                serie = pd.Series(importancias).sort_values().tail(8)
                etiquetas = [NOMBRES.get(k, k) for k in serie.index]
                porcentajes = serie.values * 100
                fig3, ax3 = plt.subplots(figsize=(5, 3.4))
                ax3.barh(etiquetas, porcentajes, color="#2f5496")
                ax3.set_xlabel("Importancia (%)")
                ax3.set_xlim(0, max(porcentajes) * 1.25)
                for i, v in enumerate(porcentajes):
                    ax3.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

                top = [NOMBRES.get(k, k) for k in list(serie.index)[::-1][:3]]
                st.write(f"Los datos que más pesan son **{top[0]}**, **{top[1]}** y "
                         f"**{top[2]}**. Un porcentaje alto significa que ese dato ayudó más a "
                         "separar a los pacientes sanos de los enfermos. No significa que "
                         "sea la causa de la enfermedad.")
            else:
                st.info("Presiona '🚀 Entrenar modelo' para ver este gráfico.")

# ===== PESTAÑA 3: PREDICCIÓN =====
with tab_pred:
    st.subheader("Probar un paciente nuevo")
    st.write("Ajusta los datos con los controles, o carga un ejemplo, y presiona "
             "**Realizar predicción**.")

    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    b1.button("Cargar ejemplo A", on_click=cargar_ejemplo, args=(EJEMPLO_A,))
    b2.button("Cargar ejemplo B", on_click=cargar_ejemplo, args=(EJEMPLO_B,))
    b3.button("Restablecer", on_click=cargar_ejemplo, args=(VALORES_INICIALES,))

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("**Datos generales**")
            st.slider("Edad (años)", 20, 100, key="edad")
            st.radio("Sexo", [0, 1], format_func=SEXO.get, horizontal=True, key="sexo")
            st.slider("Presión en reposo (mm Hg)", 80, 220, key="presion")
            st.slider("Colesterol (mg/dl)", 100, 600, key="colesterol")
            st.radio("Azúcar en ayunas mayor a 120 mg/dl", [0, 1],
                     format_func=SINO.get, horizontal=True, key="azucar")

    with col2:
        with st.container(border=True):
            st.markdown("**Síntomas y esfuerzo**")
            st.selectbox("Tipo de dolor de pecho", [1, 2, 3, 4],
                         format_func=DOLOR.get, key="tipo_dolor")
            st.slider("Pulso máximo alcanzado", 60, 220, key="pulso_max")
            st.radio("Angina por ejercicio", [0, 1],
                     format_func=SINO.get, horizontal=True, key="angina")
            st.slider("Depresión ST", 0.0, 7.0, step=0.1, key="depresion_st")

    with col3:
        with st.container(border=True):
            st.markdown("**Exámenes del corazón**")
            st.selectbox("Electrocardiograma en reposo", [0, 1, 2],
                         format_func=ECG.get, key="ecg")
            st.selectbox("Pendiente del segmento ST", [1, 2, 3],
                         format_func=PENDIENTE.get, key="pendiente")
            st.radio("Vasos principales coloreados", [0, 1, 2, 3],
                     horizontal=True, key="vasos")
            st.selectbox("Talasemia", [3, 6, 7],
                         format_func=TALASEMIA.get, key="talasemia")

    if st.button("Realizar predicción", type="primary"):
        paciente = {clave: st.session_state[clave] for clave in VALORES_INICIALES}
        with st.spinner("Consultando al modelo..."):
            respuesta = llamar_api("/predecir", paciente)
        if respuesta is None:
            st.session_state["resultado"] = None
            st.error("No se pudo obtener la predicción. Revisa que la API esté encendida.")
        else:
            st.session_state["resultado"] = {"respuesta": respuesta, "paciente": paciente}

    # Mostrar el resultado guardado
    res = st.session_state["resultado"]
    if res is not None:
        prob = res["respuesta"]["probabilidad_enfermedad"]
        alto = res["respuesta"]["diagnostico"] == 1

        st.divider()
        st.subheader("Resultado")
        r1, r2 = st.columns(2)

        with r1:
            figm = dibujar_medidor(prob)
            st.pyplot(figm)
            plt.close(figm)

        with r2:
            if alto:
                st.error("### Riesgo ALTO de enfermedad cardíaca")
                st.write("El modelo encontró en estos datos un patrón parecido al de los "
                         "pacientes enfermos del conjunto de datos. Lo recomendable es "
                         "consultar con un profesional de la salud.")
            else:
                st.success("### Riesgo BAJO de enfermedad cardíaca")
                st.write("El modelo encontró en estos datos un patrón parecido al de los "
                         "pacientes sanos del conjunto de datos. Aun así, los chequeos "
                         "médicos periódicos siguen siendo importantes.")
            st.metric("Probabilidad estimada de enfermedad", f"{prob:.1%}")

        with st.expander("Ver los datos ingresados"):
            tabla = pd.DataFrame({
                "Dato": [NOMBRES[k] for k in res["paciente"]],
                "Valor": [str(TEXTOS[k].get(v, v)) if k in TEXTOS else str(v)
                          for k, v in res["paciente"].items()],
            })
            st.dataframe(tabla, hide_index=True)

        st.caption("Este resultado es una estimación educativa y no reemplaza un "
                   "diagnóstico médico profesional.")