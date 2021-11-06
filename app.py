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

st.set_page_config(layout="wide")


matplotlib.use('Agg')


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


################################################
#          PREPROCESADO                        #
################################################

@st.cache
def load_data():

    n_trials = 30

    # ID del estudio
    id_trial = range(1, n_trials+1)
    # Gravedad de la enfermedad
    gravedad = np.random.choice([0, 1], size=n_trials, p=[0.7, 0.3])
    # Duración (semanas)
    duracion = np.random.normal(loc=20, scale=5, size=n_trials).astype('int')
    # Num. extracciones
    num_extracciones = (np.random.uniform(low=0.25, high=2, size=n_trials) * duracion).astype('int')
    # presencialidad 
    presencialidad = np.random.choice([0.25, 0.5, 0.75, 1.], size=n_trials, p=[0.1, 0.2, 0.3, 0.4])
    # Num. dosis por semana
    dosis_semana = np.random.uniform(low=4, high=14, size=n_trials).astype('int')
    # Num. pacientes
    num_pacientes = np.random.randint(low=50, high=120, size=n_trials)

    df = pd.DataFrame(data={'id_trial': id_trial,
                            'gravedad': gravedad,
                           'duracion': duracion,
                           'num_extracciones': num_extracciones,
                            'dosis_semana': dosis_semana,
                           'presencialidad': presencialidad,
                           'num_pacientes': num_pacientes})
    df = df.loc[df.index.repeat(df['num_pacientes'])].reset_index(drop=True)

    n = df.shape[0]

    # edad del paciente
    df['edad'] = np.random.normal(loc=50, scale=10, size=n).astype('int')
    df['sexo_M'] = np.random.choice([0, 1], size=n)
    df['ocupacion'] = ['Trabajando' if p > 0.2 else 'Parado' for p in np.random.random(size=n)]

    df.loc[df['edad'] > 65,'ocupacion'] = 'Jubilado' 
    df.loc[df['edad'] < 18, 'ocupacion'] = 'Escolarizado'

    df['leaves_trial'] = 0.25 * df['edad'] + 2 * df['sexo_M'] - 5 * df['gravedad'] + 5 * df['presencialidad'] +  \
    0.4 * df['num_extracciones'] 

    for idx, row in df.iterrows():
        if row['ocupacion'] == 'Jubilado':
            row['leaves_trial'] -= 10
        if row['ocupacion'] == 'Trabajando':
            row['leaves_trial'] += 5
        if row['ocupacion'] == 'Parado':
            row['leaves_trial'] -= 5
        if row['ocupacion'] == 'Escolarizado':
            row['leaves_trial'] -= 5


    df['leaves_trial'] = [1 if value > 28. else 0 for value in df['leaves_trial']]

    X, y = df.drop(columns=['id_trial', 'num_pacientes', 'leaves_trial']), df['leaves_trial']

    ocupacion_label = X['ocupacion'].values.reshape(-1, 1)
    ohe = OneHotEncoder(categories=[['Escolarizado', 'Trabajando', 'Parado', 'Jubilado']],
        drop='first', sparse=False)
    ocupacion_ohe = ohe.fit_transform(ocupacion_label)
    
    ocupacion_ohe = pd.DataFrame(data=ocupacion_ohe, columns=ohe.categories_[0][1:])
    X = pd.concat([X, ocupacion_ohe], axis=1)
    X = X.drop(columns='ocupacion')
    return X, y

########################################
#             APP                      #
########################################

st.title("DataXplain")

X, y = load_data()

# train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X, y)

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
        st.dataframe(X_single)
        shap_values = explainer(X_single)

with col2:
    st.header("Predicción")
    fig = shap.plots.waterfall(shap_values[0], max_display=9)
    st.pyplot(fig=fig, bbox_inches='tight')
    plt.clf()
