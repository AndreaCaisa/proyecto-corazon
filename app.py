import streamlit as st
import requests
import matplotlib.pyplot as plt

# Dirección de la API
API = "http://127.0.0.1:8000"

# Configuración general de la página
st.set_page_config(page_title="Riesgo de enfermedad cardíaca", layout="wide")


# ---------- FUNCIONES AUXILIARES ----------
def llamar_api(ruta, datos):
    # Envía datos a la API y devuelve la respuesta (o None si falla)
    try:
        respuesta = requests.post(API + ruta, json=datos, timeout=60)
        respuesta.raise_for_status()
        return respuesta.json()
    except Exception:
        return None


def api_disponible():
    # Revisa si la API está encendida
    try:
        requests.get(API + "/metricas", timeout=3)
        return True
    except Exception:
        return False


# Memoria de la página: así los resultados no se borran al hacer clic
if "metricas" not in st.session_state:
    st.session_state["metricas"] = None
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

# ---------- ENCABEZADO ----------
st.title("Sistema de predicción de riesgo de enfermedad cardíaca")
st.write("Aplicación de Machine Learning con Random Forest y datos clínicos reales.")

# ---------- BARRA LATERAL: PARÁMETROS Y ENTRENAMIENTO ----------
st.sidebar.header("Configuración del modelo")

n_arboles = st.sidebar.slider("Número de árboles", 10, 300, 100)
profundidad = st.sidebar.slider("Profundidad máxima", 1, 15, 5)

if st.sidebar.button("Entrenar modelo", type="primary"):
    with st.spinner("Entrenando el modelo..."):
        st.session_state["metricas"] = llamar_api(
            "/entrenar", {"n_arboles": n_arboles, "profundidad": profundidad}
        )
    if st.session_state["metricas"] is None:
        st.sidebar.error("No se pudo entrenar. Revisa que la API esté encendida.")
    else:
        st.sidebar.success("Modelo entrenado correctamente.")

# Estado de la API
if api_disponible():
    st.sidebar.write("Estado de la API: conectada")
else:
    st.sidebar.write("Estado de la API: desconectada")

# ---------- PESTAÑAS ----------
tab1, tab2, tab3 = st.tabs(["Resultados del entrenamiento", "Predicción de un paciente", "Acerca del proyecto"])

# ===== PESTAÑA 1: MÉTRICAS =====
with tab1:
    metricas = st.session_state["metricas"]

    if metricas is None:
        st.info("Configura los parámetros en la barra lateral y presiona 'Entrenar modelo'.")
    else:
        st.subheader("Métricas del modelo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exactitud", f"{metricas['exactitud']:.1%}")
        c2.metric("Precisión", f"{metricas['precision']:.1%}")
        c3.metric("Recall", f"{metricas['recall']:.1%}")
        c4.metric("F1", f"{metricas['f1']:.1%}")

        st.subheader("Matriz de confusión")
        matriz = metricas["matriz"]

        # Dibujar la matriz como un gráfico
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(matriz, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sano", "Enfermo"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sano", "Enfermo"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Diagnóstico real")

        # Escribir el número dentro de cada casilla
        maximo = max(max(fila) for fila in matriz)
        for i in range(2):
            for j in range(2):
                color = "white" if matriz[i][j] > maximo / 2 else "black"
                ax.text(j, i, matriz[i][j], ha="center", va="center",
                        fontsize=14, color=color)

        col_grafico, col_texto = st.columns([1, 1])
        with col_grafico:
            st.pyplot(fig)
        with col_texto:
            st.write(f"Pacientes enfermos que el modelo no detectó: **{matriz[1][0]}**")
            st.write(f"Pacientes sanos marcados como enfermos: **{matriz[0][1]}**")
            st.caption("En salud, el recall es muy importante: indica qué porcentaje "
                       "de los pacientes enfermos logra detectar el modelo.")

# ===== PESTAÑA 2: PREDICCIÓN =====
with tab2:
    st.subheader("Datos del paciente")

    # El formulario envía todo junto al presionar el botón
    with st.form("formulario_paciente"):
        col1, col2, col3 = st.columns(3)

        with col1:
            edad = st.number_input("Edad (años)", 20, 100, 55)
            sexo = st.selectbox("Sexo", [0, 1],
                                format_func=lambda x: "Mujer" if x == 0 else "Hombre")
            tipo_dolor = st.selectbox("Tipo de dolor de pecho", [1, 2, 3, 4],
                                      format_func=lambda x: {1: "Angina típica", 2: "Angina atípica",
                                                             3: "Dolor no anginoso", 4: "Sin síntomas"}[x])
            presion = st.number_input("Presión en reposo (mm Hg)", 80, 220, 130)
            colesterol = st.number_input("Colesterol (mg/dl)", 100, 600, 240)

        with col2:
            azucar = st.selectbox("Azúcar en ayunas mayor a 120 mg/dl", [0, 1],
                                  format_func=lambda x: "No" if x == 0 else "Sí")
            ecg = st.selectbox("Electrocardiograma en reposo", [0, 1, 2],
                               format_func=lambda x: {0: "Normal", 1: "Anomalía ST-T", 2: "Hipertrofia"}[x])
            pulso_max = st.number_input("Pulso máximo alcanzado", 60, 220, 150)
            angina = st.selectbox("Angina por ejercicio", [0, 1],
                                  format_func=lambda x: "No" if x == 0 else "Sí")

        with col3:
            depresion_st = st.number_input("Depresión ST", 0.0, 7.0, 1.0, step=0.1)
            pendiente = st.selectbox("Pendiente del segmento ST", [1, 2, 3],
                                     format_func=lambda x: {1: "Ascendente", 2: "Plana", 3: "Descendente"}[x])
            vasos = st.selectbox("Vasos principales coloreados", [0, 1, 2, 3])
            talasemia = st.selectbox("Talasemia", [3, 6, 7],
                                     format_func=lambda x: {3: "Normal", 6: "Defecto fijo",
                                                            7: "Defecto reversible"}[x])

        enviado = st.form_submit_button("Realizar predicción", type="primary")

    # Cuando se presiona el botón, se consulta a la API
    if enviado:
        paciente = {
            "edad": edad, "sexo": sexo, "tipo_dolor": tipo_dolor, "presion": presion,
            "colesterol": colesterol, "azucar": azucar, "ecg": ecg, "pulso_max": pulso_max,
            "angina": angina, "depresion_st": depresion_st, "pendiente": pendiente,
            "vasos": vasos, "talasemia": talasemia,
        }
        st.session_state["resultado"] = llamar_api("/predecir", paciente)
        if st.session_state["resultado"] is None:
            st.error("No se pudo obtener la predicción. Revisa que la API esté encendida.")

    # Mostrar el resultado guardado
    resultado = st.session_state["resultado"]
    if resultado is not None:
        st.subheader("Resultado")
        probabilidad = resultado["probabilidad_enfermedad"]

        if resultado["diagnostico"] == 1:
            st.error(f"Riesgo ALTO de enfermedad cardíaca. Probabilidad estimada: {probabilidad:.1%}")
        else:
            st.success(f"Riesgo BAJO de enfermedad cardíaca. Probabilidad estimada: {probabilidad:.1%}")

        st.progress(probabilidad)
        st.caption("Este resultado es una estimación educativa y no reemplaza "
                   "un diagnóstico médico profesional.")

# ===== PESTAÑA 3: INFORMACIÓN =====
with tab3:
    st.subheader("Descripción del proyecto")
    st.write("Aplicación que entrena un modelo Random Forest para estimar el riesgo de "
             "enfermedad cardíaca a partir de 13 variables clínicas del paciente.")

    st.subheader("Datos utilizados")
    st.write("Dataset Heart Disease (versión Cleveland) con 303 pacientes reales.")
    st.caption("Fuente: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). "
               "Heart Disease [Dataset]. UCI Machine Learning Repository. "
               "https://doi.org/10.24432/C52P4X")

    st.subheader("Cómo usar la aplicación")
    st.write("1. Ajusta los parámetros en la barra lateral y entrena el modelo.")
    st.write("2. Revisa las métricas en la pestaña de resultados.")
    st.write("3. Ingresa los datos de un paciente en la pestaña de predicción.")