import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import io 
import plotly.express as px     

def app():
    st.markdown("## 📎📎Carga tu archivo CSV o Excel (Maximo 200MB)📎📎")
    st.markdown('''En esta sección se guardará tu dataframe para posteriormente utilizarlo en los diferentes
    métodos y algortimos que contamos en la aplicación.''') 
    st.write("\n")
    uploaded_file = st.file_uploader(label="Sube tu archivo", type=['csv','xlsx'])
    global data, fig, numeric_columns
    global s
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    
    if st.button("Cargar Datos"):
        st.write(data)
        data.to_csv('data/main_data.csv', index=False)
        st.markdown('''A continuación se describen los diferentes módulos con los que contamos, es de suma importancia
        seguir dichos módulos en orden, esto para evitar algun error en los algortimos o métodos de análisis''')

        st.subheader('📌  **1.- Carga de datos**')
        st.caption('''Este módulo se ocupa de la carga de datos. Puede tomar archivos csv y excel. Tan pronto como se cargan los datos, 
        crea una copia de los datos para garantizar que no tengamos que leer los datos varias veces. ''')

        st.subheader('📌  **2.- Selección de Caracteristicas**')
        st.caption(''' Este módulo se encarga de mostrarle parte del Análisis Exploratorio de Datos (EDA), esto para que 
        el usuario tenga una visión mas detallada y precisa de las variables a analizar. Ayuda a estudiar la tendencia, 
        distribución y forma de cada una de las variables.''')
        st.caption('''Por otro lado La selección de características hace referencia al proceso de reducir las entradas para su 
        procesamiento y análisis, o de encontrar las entradas más significativas y reducir la dimensionalidad. Una vez que haya
        realizado el EDA y seleccione sus variables más significativas, usted podra eliminarlas del dataframe original''')

        st.subheader('📌  **3.- Análsis de Componentes Principales (PCA)**')
        st.caption('''Este módulo presenta el PCA para ayudarnos a perder la menor cantidad de información (varianza) posible. 
        Cuando contamos con un gran número de variables PCA nos permite reducirlas a un número menor de variables transformadas 
        (componentes principales) que expliquen gran parte de la variabilidad en los datos. Una vez teniendo sus componentes principales,
        procedemos a eliminar del dataframe los datos que no se usaran para usar el nuevo dataframe en los siguientes pasos. ''')
        st.caption('''Procedimiento:\n
        1.- Se hace una estandarización de los datos.\n
        2.- A partir de los datos estandarizados, se calcula una matriz de covarianzas o correlaciones.\n
        3.- Se calculan los componentes (eigen-vectores) y la varianza (eigen-valores) a partir de la matriz anterior.\n
        4.- Se decide el número de componentes principales.\n
        5.- Se examina la proporción de relevancia (cargas)''')

        st.subheader('📌  **4.- Métricas de Distancia**')
        st.caption(''' Este módulo nos permitirá calcular la distancia entre dos elementos y asi indicarnos que tan 
        difrenetes o similares son. Existen varias formas de medir la distancia entre dos elementos, y elegir la métrica adecuada para cada problema es un paso crucial para
        obtener buenos resultados en cualquier aplicación de minería de datos. En este apartado usted podra averiguar que 
        métricas estan disponibles.''')

        st.subheader('📌  **5.- Clusterización**')
        st.caption(''' A partir del dataframe generado en el paso 3 (PCA), este módulo le ayudará a, que a partir de un conjunto de
        datos, tratar de obtener grupos de objetos, de tal manera que los objetos que pertenecen a un grupo sean muy homogéneos entre sí y 
        por otra parte la heterogeneidad entre los distintos grupos sea muy elevada. El Clustering se enmarca dentro del aprendizaje no supervisado; 
        es decir, que para esta técnica solo disponemos de un conjunto de datos de entrada, sobre los que debemos obtener información sobre la 
        estructura del dominio de salida, que es una información de la cual no se dispone.''' )
        st.caption('''Datos listos pone a su disposición el Clustering Jerárquico y Clustering Particional. Dirigase al apartado de
        Clusterización para obtener más detalle.''')

        st.subheader('📌  **6.- Pronóstico con Regresión Lineal**')
        st.caption('''La regresión es una forma estadística de establecer una relación entre una variable dependiente y un conjunto de variables independientes. 
        Es un enfoque muy simple para el aprendizaje supervisado, su propósito es establecer un modelo para la relación entre un cierto número de características 
        y una variable objetivo continua, calcula una ecuación que minimiza la distancia entre la línea ajustada (recta) y todos los puntos de datos.''')

        st.subheader('📌  **7.- Árboles de desición**')
        st.caption('''Este módulo nos ayuda a generar árbol de desiciones, algoritmos estadísticos o técnicas de machine learning que nos permiten la construcción
        de modelos predictivos de analítica de datos basados en su clasificación según ciertas características o propiedades, o en la regresión mediante la relación 
        entre distintas variables para predecir el valor de otra.\n
        Estructura:\n
        • Nodo principal. Representa toda la población (todos los datos) que posteriormente se dividirá. También cuenta como un nodo de decisión.\n
        • Nodo de decisión. Se encarga de dividir los datos, dependiendo de una decisión. Se tomandos caminos.\n
        • Nodo hoja. Es donde recae la decisión final.\n
        • Profundidad. La profundidad indica los niveles que tiene el árbol de decisión.''')
        st.caption('''Datos listos pone a su disposición el Arbol de desición Pronóstico  y Arbol de desición Clasificación. Dirigase al apartado de
        Clusterización para obtener más detalle.''')

        st.subheader('📌  **8.- Bosques Aleatorios**')
        st.caption('''Usar árboles tienen una gran desventaja: el overfitting. Esto quiere decir que en general funcionan muy bien durante el entrenamiento, 
        pero no tanto cuando introducimos datos nuevos (es decir cuándo queremos hacer predicciones). Y es esto dio origen a los Bosques Aleatorios (Random Forests), 
        uno de los algoritmos más poderosos y más usados del Machine Learning. En lugar de entrenar un único árbol se entrenan varios, usualmente decenas o cientos. Es
        importante mencionar que cada árbol del bosque se entrena con una parte distinta del dataset original.''')
        st.caption('''Datos listos pone a su disposición el Bosques aleatorios Pronóstico  y Bosques aleatorios Clasificación. Dirigase al apartado de
        Clusterización para obtener más detalle.''')


