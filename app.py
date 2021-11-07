import joblib
import re

import shap
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)


im = Image.open("logo.png")
st.set_page_config(
    page_title="DataXplain",
    page_icon=im,
    layout="wide",
)


################################################
#          PREPROCESADO                        #
################################################

def shap_transform_scale(original_shap_values, Y_pred, which):
    from scipy.special import expit
    
    #Compute the transformed base value, which consists in applying the logit function to the base value
    from scipy.special import expit #Importing the logit function for the base value transformation
    untransformed_base_value = original_shap_values.base_values[-1]
   
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]
    
    base_value = expit(untransformed_base_value) # = 1 / (1+ np.exp(-untransformed_base_value))

    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = Y_pred[which] - base_value

    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain

    #Transforming the original shapley values to the new scale
    shap_values_transformed = original_shap_values / distance_coefficient

    #Finally resetting the base_value as it does not need to be transformed
    base_value_array = np.ones(original_shap_values.base_values.shape)*base_value
    shap_values_transformed.base_values = base_value_array
    shap_values_transformed.data = original_shap_values.data
    
    #Now returning the transformed array
    return shap_values_transformed    

@st.cache
def load_data():

    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    
    return X, y

########################################
#             APP                      #
########################################

#st.title("DataXplain")
st.image('titulo.png')

X, y = load_data()

# load XGBoost model
model = joblib.load('model.joblib').fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# predict class and probabilities
res = model.predict(X)
proba = model.predict_proba(X)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Introducción de datos")
    form = st.form(key='my_form')
    gravedad = form.number_input("Gravedad de la enfermedad a estudiar (0=leve, 1=grave)", min_value=0, max_value=1, value=0)
    duracion = form.number_input("Duración del estudio (en semanas)", min_value=1, max_value=100, value=18)
    num_extracciones = form.number_input("Número de extracciones totales", min_value=0, max_value=100, value=8)
    dosis_semana = form.number_input("Dosis semanales que ha de consumir el paciente", min_value=0, max_value=100, value=8)
    presencialidad = form.number_input("Porcentaje de sesiones presenciales (%)", min_value=0, max_value=100, value=75) / 100.
    edad = form.number_input("Edad del paciente", min_value=1, max_value=120, value=61)
    sexo = form.selectbox("Sexo", ("Mujer", "Hombre"))
    sexo = 1 if sexo == 'Hombre' else 0
    ocupacion = form.selectbox("Ocupación", ("Escolarizado", "Trabajando", "Parado", "Jubilado"))

    submit_button = form.form_submit_button(label='Predecir')

    if submit_button:
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
        # predict class and probabilities
        res = model.predict(X_single)
        proba = model.predict_proba(X_single)

with col2:
    st.header("Predicción")
    
    # Cambiamos los nombres
    shap_values.feature_names = ['Gravedad (0->leve 1->grave)',
                                 'Duracion (semanas)',
                                 '# Extracciones',
                                 'Dosis semanales',
                                 '% Presencialidad',
                                 'Edad',
                                 'Sexo (0->Mujer, 1->Hombre)',
                                 'Trabajando',
                                 'Parado',
                                 'Jubilado']

    # Transform from logits to probabilities
    shap_value_transformed = shap_transform_scale(shap_values, proba[:, 1], 0)
    shap.plots.waterfall(shap_value_transformed[0], max_display=9, show=True)
    fig = plt.gcf() # gcf means "get current figure"
    ax = plt.gca() #gca means "get current axes"
    ax, ax2, ax3 = fig.axes
    
    # Convert from decimal to probabilities (%)
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            text = child.get_text()
            # if the string has a +/- at the start
            if text.startswith('−') or text.startswith('+'):
                # extract float and multiply by 100 to get probability
                try:
                    num = float(re.findall(r'\d+\.\d+', text)[0])*100
                except IndexError:
                    num = float(re.findall(r'\d+', text)[0])*100
                # modify the text
                child.set_text(text[0] + '{:.2f}'.format(num) + '%')
    
    pred_str = 'NO ABANDONA' if res[0] == 0 else 'ABANDONA'
    
    fig.suptitle('PREDICCIÓN: ' + pred_str, x=0.5, y=1.05)
    
    # Modificamos el nombre de las etiquetas
    labels_ax2 = [item.get_text() for item in ax2.xaxis.get_majorticklabels()]
    labels_ax2[0] = "\n$Probabilidad \,\, base$"
    try:
        prob_base = float(re.findall(r'\d+\.\d+', labels_ax2[1])[0])*100
    except:
        prob_base = float(re.findall(r'\d+', labels_ax2[1])[0])*100

    labels_ax2[1] = '\n$ = ' + '{:.2f}'.format(prob_base) +'\%$'
    
    labels_ax3 = [item.get_text() for item in ax3.xaxis.get_majorticklabels()]
    labels_ax3[0] = "$Probabilidad \,\, final$"
    try:
        prob_final = float(re.findall(r'\d+\.\d+', labels_ax3[1])[0])*100
    except:
        prob_final = float(re.findall(r'\d+', labels_ax3[1])[0])*100

    labels_ax3[1] = '\n$ = ' + '{:.2f}'.format(prob_final) +'\%$'

    ax2.set_xticklabels(labels_ax2, fontsize=12, ha="left")
    ax3.set_xticklabels(labels_ax3, fontsize=12, ha="left")

    # Modificamos ligeramente la posición de los números

    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-1.4, 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(-18/72., 0, fig.dpi_scale_trans))

    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-1.3, -1/72., fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(-26/72., -1/72., fig.dpi_scale_trans))
    st.pyplot(fig=fig, bbox_inches='tight')
    plt.clf()
