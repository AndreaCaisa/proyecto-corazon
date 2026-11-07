import pandas as pd
import matplotlib.pyplot as plt

# Dirección del dataset real (UCI, versión Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Nombres de las columnas (el archivo original no las trae)
columnas = ["edad", "sexo", "tipo_dolor", "presion", "colesterol",
            "azucar", "ecg", "pulso_max", "angina", "depresion_st",
            "pendiente", "vasos", "talasemia", "diagnostico"]

# Leer el archivo (los "?" son datos faltantes)
datos = pd.read_csv(url, names=columnas, na_values="?")

# Convertir el diagnóstico: 0 = sano, 1 = enfermedad (antes iba de 0 a 4)
datos["diagnostico"] = (datos["diagnostico"] > 0).astype(int)

# Mostrar información básica
print("Filas y columnas:", datos.shape)
print(datos.head())
print("\nDatos faltantes por columna:")
print(datos.isnull().sum())

# Guardar una copia local para usarla después
datos.to_csv("corazon.csv", index=False)

# Gráfica: cuántos sanos y cuántos enfermos hay
datos["diagnostico"].value_counts().plot(kind="bar")
plt.title("0 = sano, 1 = enfermedad")
plt.show()

