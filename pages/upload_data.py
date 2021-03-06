import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import io 
import plotly.express as px     

def app():
    st.markdown("## 馃搸馃搸Carga tu archivo CSV o Excel (Maximo 200MB)馃搸馃搸")
    st.markdown('''En esta secci贸n se guardar谩 tu dataframe para posteriormente utilizarlo en los diferentes
    m茅todos y algortimos que contamos en la aplicaci贸n.''') 
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
        st.markdown('''A continuaci贸n se describen los diferentes m贸dulos con los que contamos, es de suma importancia
        seguir dichos m贸dulos en orden, esto para evitar algun error en los algortimos o m茅todos de an谩lisis''')

        st.subheader('馃搶  **1.- Carga de datos**')
        st.caption('''Este m贸dulo se ocupa de la carga de datos. Puede tomar archivos csv y excel. Tan pronto como se cargan los datos, 
        crea una copia de los datos para garantizar que no tengamos que leer los datos varias veces. ''')

        st.subheader('馃搶  **2.- Selecci贸n de Caracteristicas**')
        st.caption(''' Este m贸dulo se encarga de mostrarle parte del An谩lisis Exploratorio de Datos (EDA), esto para que 
        el usuario tenga una visi贸n mas detallada y precisa de las variables a analizar. Ayuda a estudiar la tendencia, 
        distribuci贸n y forma de cada una de las variables.''')
        st.caption('''Por otro lado La selecci贸n de caracter铆sticas hace referencia al proceso de reducir las entradas para su 
        procesamiento y an谩lisis, o de encontrar las entradas m谩s significativas y reducir la dimensionalidad. Una vez que haya
        realizado el EDA y seleccione sus variables m谩s significativas, usted podra eliminarlas del dataframe original''')

        st.subheader('馃搶  **3.- An谩lsis de Componentes Principales (PCA)**')
        st.caption('''Este m贸dulo presenta el PCA para ayudarnos a perder la menor cantidad de informaci贸n (varianza) posible. 
        Cuando contamos con un gran n煤mero de variables PCA nos permite reducirlas a un n煤mero menor de variables transformadas 
        (componentes principales) que expliquen gran parte de la variabilidad en los datos. Una vez teniendo sus componentes principales,
        procedemos a eliminar del dataframe los datos que no se usaran para usar el nuevo dataframe en los siguientes pasos. ''')
        st.caption('''Procedimiento:\n
        1.- Se hace una estandarizaci贸n de los datos.\n
        2.- A partir de los datos estandarizados, se calcula una matriz de covarianzas o correlaciones.\n
        3.- Se calculan los componentes (eigen-vectores) y la varianza (eigen-valores) a partir de la matriz anterior.\n
        4.- Se decide el n煤mero de componentes principales.\n
        5.- Se examina la proporci贸n de relevancia (cargas)''')

        st.subheader('馃搶  **4.- M茅tricas de Distancia**')
        st.caption(''' Este m贸dulo nos permitir谩 calcular la distancia entre dos elementos y asi indicarnos que tan 
        difrenetes o similares son. Existen varias formas de medir la distancia entre dos elementos, y elegir la m茅trica adecuada para cada problema es un paso crucial para
        obtener buenos resultados en cualquier aplicaci贸n de miner铆a de datos. En este apartado usted podra averiguar que 
        m茅tricas estan disponibles.''')

        st.subheader('馃搶  **5.- Clusterizaci贸n**')
        st.caption(''' A partir del dataframe generado en el paso 3 (PCA), este m贸dulo le ayudar谩 a, que a partir de un conjunto de
        datos, tratar de obtener grupos de objetos, de tal manera que los objetos que pertenecen a un grupo sean muy homog茅neos entre s铆 y 
        por otra parte la heterogeneidad entre los distintos grupos sea muy elevada. El Clustering se enmarca dentro del aprendizaje no supervisado; 
        es decir, que para esta t茅cnica solo disponemos de un conjunto de datos de entrada, sobre los que debemos obtener informaci贸n sobre la 
        estructura del dominio de salida, que es una informaci贸n de la cual no se dispone.''' )
        st.caption('''Datos listos pone a su disposici贸n el Clustering Jer谩rquico y Clustering Particional. Dirigase al apartado de
        Clusterizaci贸n para obtener m谩s detalle.''')

        st.subheader('馃搶  **6.- Pron贸stico con Regresi贸n Lineal**')
        st.caption('''La regresi贸n es una forma estad铆stica de establecer una relaci贸n entre una variable dependiente y un conjunto de variables independientes. 
        Es un enfoque muy simple para el aprendizaje supervisado, su prop贸sito es establecer un modelo para la relaci贸n entre un cierto n煤mero de caracter铆sticas 
        y una variable objetivo continua, calcula una ecuaci贸n que minimiza la distancia entre la l铆nea ajustada (recta) y todos los puntos de datos.''')

        st.subheader('馃搶  **7.- 脕rboles de desici贸n**')
        st.caption('''Este m贸dulo nos ayuda a generar 谩rbol de desiciones, algoritmos estad铆sticos o t茅cnicas de machine learning que nos permiten la construcci贸n
        de modelos predictivos de anal铆tica de datos basados en su clasificaci贸n seg煤n ciertas caracter铆sticas o propiedades, o en la regresi贸n mediante la relaci贸n 
        entre distintas variables para predecir el valor de otra.\n
        Estructura:\n
        鈥? Nodo principal. Representa toda la poblaci贸n (todos los datos) que posteriormente se dividir谩. Tambi茅n cuenta como un nodo de decisi贸n.\n
        鈥? Nodo de decisi贸n. Se encarga de dividir los datos, dependiendo de una decisi贸n. Se tomandos caminos.\n
        鈥? Nodo hoja. Es donde recae la decisi贸n final.\n
        鈥? Profundidad. La profundidad indica los niveles que tiene el 谩rbol de decisi贸n.''')
        st.caption('''Datos listos pone a su disposici贸n el Arbol de desici贸n Pron贸stico  y Arbol de desici贸n Clasificaci贸n. Dirigase al apartado de
        Clusterizaci贸n para obtener m谩s detalle.''')

        st.subheader('馃搶  **8.- Bosques Aleatorios**')
        st.caption('''Usar 谩rboles tienen una gran desventaja: el overfitting. Esto quiere decir que en general funcionan muy bien durante el entrenamiento, 
        pero no tanto cuando introducimos datos nuevos (es decir cu谩ndo queremos hacer predicciones). Y es esto dio origen a los Bosques Aleatorios (Random Forests), 
        uno de los algoritmos m谩s poderosos y m谩s usados del Machine Learning. En lugar de entrenar un 煤nico 谩rbol se entrenan varios, usualmente decenas o cientos. Es
        importante mencionar que cada 谩rbol del bosque se entrena con una parte distinta del dataset original.''')
        st.caption('''Datos listos pone a su disposici贸n el Bosques aleatorios Pron贸stico  y Bosques aleatorios Clasificaci贸n. Dirigase al apartado de
        Clusterizaci贸n para obtener m谩s detalle.''')


