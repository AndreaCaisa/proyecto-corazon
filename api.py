from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from modelo import entrenar_modelo

app = FastAPI()

# Aquí se guardan las últimas métricas del entrenamiento
ultimas_metricas = {}

# Parámetros que se pueden configurar para entrenar
class Parametros(BaseModel):
    n_arboles: int = 100
    profundidad: int = 5

# Datos de un paciente nuevo
class Paciente(BaseModel):
    edad: float
    sexo: float
    tipo_dolor: float
    presion: float
    colesterol: float
    azucar: float
    ecg: float
    pulso_max: float
    angina: float
    depresion_st: float
    pendiente: float
    vasos: float
    talasemia: float

# Función 1: entrenar el modelo con los parámetros recibidos
@app.post("/entrenar")
def entrenar(parametros: Parametros):
    global ultimas_metricas
    ultimas_metricas = entrenar_modelo(parametros.n_arboles, parametros.profundidad)
    return ultimas_metricas

# Función 2: ver las métricas del último entrenamiento
@app.get("/metricas")
def metricas():
    if not ultimas_metricas:
        return {"mensaje": "Primero entrena el modelo"}
    return ultimas_metricas

# Función 3: predecir un caso nuevo
@app.post("/predecir")
def predecir(paciente: Paciente):
    modelo = joblib.load("modelo.joblib")
    datos = pd.DataFrame([paciente.model_dump()])
    resultado = int(modelo.predict(datos)[0])
    probabilidad = float(modelo.predict_proba(datos)[0][1])
    return {"diagnostico": resultado, "probabilidad_enfermedad": probabilidad}