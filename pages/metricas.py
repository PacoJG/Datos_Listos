import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import os
import plotly.express as px
from scipy.spatial.distance import cdist 
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def app():
    st.title("Métricas de Distancia")
    st.caption('''Una métrica es una función que calcula la distancia entre dos elementos y qué por tanto se 
    utiliza para medir cuán diferentes o similares son. Existen varias formas de medir la distancia entre dos 
    elementos, y elegir la métrica adecuada para cada problema es un paso crucial para obtener buenos resultados 
    en cualquier aplicación de minería de datos. A menudo se utilizan funciones que miden la similitud entre dos 
    elementos en lugar de su distancia. Estas mediciones se utilizan para “aprender de los datos” ''')
    if 'main_data_final.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data_final.csv')
        df.to_csv('data/main_data_new.csv', index=False)
        st.header("Estandarización de los datos")
        estandarizar = StandardScaler()
        estandarizar.fit(df)
        MEstandarizada = estandarizar.transform(df) 
        st.write(pd.DataFrame(MEstandarizada, columns=df.columns))

        st.header("Matriz de distancias: EUCLIDIANA")
        st.caption('''Sus bases se encuentran en la aplicación del Teorema de Pitágoras, 
        donde la distancia viene a ser la longitud de la hipotenusa.''')
        DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
        MEuclidiana = pd.DataFrame(DstEuclidiana)
        st.write(MEuclidiana.round(3))
        st.caption('''Distancia entre pares de elementos: Ingrese el numero de los vectores de los elementos, 
        de los cuales quiere obtener su distancia''')
        eu_number1 = st.number_input("Ingresa el vector 1", min_value=0, max_value=1000, key=1)
        eu_objeto1 = MEstandarizada[eu_number1]
        eu_number2 = st.number_input("Ingresa el vector 2", min_value=0, max_value=1000, key=1)
        eu_objeto2 = MEstandarizada[eu_number2]
        dstEuclidiana = distance.euclidean(eu_objeto1,eu_objeto2)
        st.markdown("La distancia entre los 2 elementos seleccionado es:")
        st.write(dstEuclidiana)

        st.header("Matriz de distancias: CHEBYSHEV")
        st.caption('''Es el valor máximo absoluto de las diferencias entre las coordenadas 
        de un par de elementos.''')
        DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
        MChebyshev = pd.DataFrame(DstChebyshev)
        st.write(MChebyshev.round(3))
        st.caption('''Distancia entre pares de elementos: Ingrese el numero de los vectores de los elementos, 
        de los cuales quiere obtener su distancia''')
        ch_number1 = st.number_input("Ingresa el vector 1", min_value=0, max_value=1000, key=2)
        ch_objeto1 = MEstandarizada[ch_number1]
        ch_number2 = st.number_input("Ingresa el vector 2", min_value=0, max_value=1000, key=2)
        ch_objeto2 = MEstandarizada[ch_number2]
        dstChebyshev = distance.chebyshev(ch_objeto1,ch_objeto2)
        st.markdown("La distancia entre los 2 elementos seleccionado es:")
        st.write(dstChebyshev)

        st.header("Matriz de distancias: MANHATTAN")
        st.caption('''La distancia de Manhattan entre dos puntos en dos dimensiones 
        es la suma de las diferencias absolutas de sus coordenadas cartesianas.''')
        DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
        MManhattan = pd.DataFrame(DstManhattan)
        st.write(MManhattan.round(3))
        st.caption('''Distancia entre pares de elementos: Ingrese el numero de los vectores de los elementos, 
        de los cuales quiere obtener su distancia''')
        ma_number1 = st.number_input("Ingresa el vector 1", min_value=0, max_value=1000, key=3)
        ma_objeto1 = MEstandarizada[ma_number1]
        ma_number2 = st.number_input("Ingresa el vector 2", min_value=0, max_value=1000, key=3)
        ma_objeto2 = MEstandarizada[ma_number2]
        dstManhattan = distance.cityblock(ma_objeto1,ma_objeto2)
        st.markdown("La distancia entre los 2 elementos seleccionado es:")
        st.write(dstManhattan)

        st.header("Matriz de distancias: MINKOWSKI")
        st.caption('''Es una distancia entre dos puntos en un espacio n- dimensional. 
        Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev.''')
        DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
        MMinkowski = pd.DataFrame(DstMinkowski)
        st.write(MMinkowski.round(3))
        st.caption('''Distancia entre pares de elementos: Ingrese el numero de los vectores de los elementos, 
        de los cuales quiere obtener su distancia''')
        mi_number1 = st.number_input("Ingresa el vector 1", min_value=0, max_value=1000, key=4)
        mi_objeto1 = MEstandarizada[mi_number1]
        mi_number2 = st.number_input("Ingresa el vector 2", min_value=0, max_value=1000, key=4)
        mi_objeto2 = MEstandarizada[mi_number2]
        dstMinkowski = distance.minkowski(mi_objeto1,mi_objeto2)
        st.markdown("La distancia entre los 2 elementos seleccionado es:")
        st.write(dstMinkowski)


