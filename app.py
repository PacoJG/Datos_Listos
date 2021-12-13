#importar librerias
import os 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import plotly.express as px
from PIL import Image
from multiapp import MultiPage
from pages import upload_data, selec_carac, pca, metricas, cluster, pronosticorl, arboles, bosques

#Instancia de app
app = MultiPage()

image = Image.open('logo.jpg')
st.image(image, use_column_width=True)
st.set_option('deprecation.showfileUploaderEncoding',False)
st.markdown(''' **Datos Listos es una herramienta basada en inteligencia artificial que puede tomar un conjunto de datos
e identificar patrones en estos, para que nuestros usuarios puedan interpretar el resultado y luego tomar una desición.
Ofrecemos una amplia variedad de algortimos y métodos que presentarán los datos de una manera clara. La aplicación se ha construido pensando en un usuario 
bastante elemental y, por tanto, es de fácil uso y comprensión **''')

st.markdown("---")

app.add_page("Cargar Datos", upload_data.app)
app.add_page("Selección de Caracteristicas", selec_carac.app)
app.add_page("Análisis de Componentes Principales", pca.app)
app.add_page("Métricas de Distancia", metricas.app)
app.add_page("Clusterización", cluster.app)
app.add_page("Pronóstico con Regresión Lineal Múltiple", pronosticorl.app)
app.add_page("Arboles de decisión", arboles.app)
app.add_page("Bosques Aleatorios", bosques.app)

app.run()

