# Predicción de riesgo de enfermedad cardíaca

Proyecto final de Proyectos de Machine Learning (2026-2).
Aplicación que usa un modelo Random Forest para estimar el riesgo de enfermedad cardíaca.

## Integrantes
- LISETH ESTEFANIA NATO PAUCAR
- ANDREA ABIGAIL CAISA CANGO
- JOSSELYN ANDREA QUIÑONEZ USHIÑA


## Datos
Dataset Heart Disease (Cleveland), 303 pacientes reales.
Fuente: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

## Cómo ejecutarlo
1. Instalar lo necesario:
   pip install pandas scikit-learn matplotlib joblib fastapi uvicorn streamlit requests
2. Descargar los datos y generar corazon.csv:
   python explorar.py
3. Encender la API:
   python -m uvicorn api:app --reload
4. En otra terminal, abrir la interfaz:
   python -m streamlit run app.py