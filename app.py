import joblib

import shap
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(layout="wide")



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


################################################
#          PREPROCESADO                        #
################################################

@st.cache
def load_data():

    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    
    return X, y

########################################
#             APP                      #
########################################

st.title("DataXplain")

X, y = load_data()

# load XGBoost model
model = joblib.load('model.joblib').fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)


col1, col2 = st.columns([1, 2])

with col1:
    st.header("Introducción de datos")
    gravedad = st.number_input("Gravedad de la enfermedad a estudiar (0=leve, 1=grave)", min_value=0, max_value=1)
    duracion = st.number_input("Duración del estudio (en semanas)", min_value=1, max_value=100)
    num_extracciones = st.number_input("Número de extracciones totales", min_value=0, max_value=100)
    dosis_semana = st.number_input("Dosis semanales que ha de consumir el paciente", min_value=0, max_value=100)
    presencialidad = st.number_input("Porcentaje de sesiones presenciales (%)", min_value=0, max_value=100) / 100.
    edad = st.number_input("Edad del paciente", min_value=1, max_value=120)
    sexo = st.selectbox("Sexo", ("Mujer", "Hombre"))
    sexo = 1 if sexo == 'Hombre' else 0
    ocupacion = st.selectbox("Ocupacion", ("Escolarizado", "Trabajando", "Parado", "Jubilado"))

    if st.button("Predecir"):
        X_single = pd.DataFrame(data={'gravedad': [gravedad],
                                      'duracion': [duracion],
                                      'num_extracciones': [num_extracciones],
                                      'dosis_semana': [dosis_semana],
                                      'presencialidad': [presencialidad],
                                      'edad': [edad],
                                      'sexo_M': [sexo],
                                      'ocupacion': [ocupacion]})
        ocupacion_label = X_single['ocupacion'].values.reshape(-1, 1)
        ohe = OneHotEncoder(categories=[['Escolarizado', 'Trabajando', 'Parado', 'Jubilado']],
        drop='first', sparse=False)
        ocupacion_ohe = ohe.fit_transform(ocupacion_label)
        ocupacion_ohe = pd.DataFrame(data=ocupacion_ohe, columns=ohe.categories_[0][1:])
        X_single = pd.concat([X_single, ocupacion_ohe], axis=1)
        X_single = X_single.drop(columns='ocupacion')
        #st.dataframe(X_single)
        shap_values = explainer(X_single)

with col2:
    st.header("Predicción")
    fig = shap.plots.waterfall(shap_values[0], max_display=9)
    st.pyplot(fig=fig, bbox_inches='tight')
    plt.clf()
