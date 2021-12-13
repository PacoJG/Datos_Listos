import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import io 
import os
import plotly.express as px
import plotly.graph_objects as go

def app():
    global numeric_columns, df
    st.title("Selección de Caracteristicas")
    st.caption('''La selección de características hace referencia al proceso de reducir las entradas para su procesamiento y 
    análisis, o de encontrar las entradas más significativas y reducir la dimensionalidad, para ello es necesario realizar el 
    Análisis Exploratorio de Datos. Esta selección de características influye demasiado en el resultado final, en el análisis 
    final.''')
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data.csv')
        st.header("Dataframe a trabajar")
        st.write(df)
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        buffer = io.StringIO() 
        df.info(buf=buffer)
        s = buffer.getvalue() 
        st.header("Descripción del Dataframe")
        st.caption('''Necesitamos saber qué tipo de datos (enteros, flotantes, etc.) vamos a analizar, 
        saber las dimensiones de la matriz nos ayuda a tener un panoráma mejor de nuestro Dataframe.''')
        st.text(s)

        st.header("Detección de valores atípicos (Gráfica de Dispersión o Histograma")
        st.caption('''Aquí se detectan los valores fuera de rango, valores que no esperábamos. 
        ¿Qué se puede hacer con estos elementos? \n
        - Corregirlos, si tenemos la información necesaria. \n
        - Eliminar el elemento, si no se tiene la información..''')
        st.markdown(''' Del lado izquierdo (sidebar) podemos selecciónar el tipo de gráfico a visualizar,
        asi como sus parámteros''')

        chart_select = st.sidebar.selectbox(
            label="Selecciona el tipo de gráfico",
            options=['Gráfico de Dispersión', 'Histograma']
        )

        if chart_select == 'Gráfico de Dispersión':
            st.sidebar.subheader("Configuración del Gráfico de Dispersión")
            x_values = st.sidebar.selectbox('X', options=numeric_columns, key="1")
            y_values = st.sidebar.selectbox('Y', options=numeric_columns, key="1")
            #color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(df, x=x_values, y=y_values)
            # display the chart
            st.plotly_chart(plot)

        if chart_select == 'Histograma':
            st.sidebar.subheader("Configuración del Histogram")
            try:
                x = st.sidebar.selectbox('Caracteristicas', options=numeric_columns, key="1")
                bin_size = st.sidebar.slider("Cantidad de contenedores", min_value=10, max_value=100, value=40)
                plot = px.histogram(df, x=x, nbins=bin_size)
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
        
        st.header("Resumen estadístico de variables numéricas")
        st.caption("Se sacan estadísticas usando describe() que muestra un resumen estadístico de las variables numéricas.")
        st.write(df.describe())
        st.header("Identificación de relaciones entre pares de variables")
        st.caption('''Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas.
        Con ayuda del operador corr() de Python, observamos el coeficiente de correlación de cada una de las variables 
        que estamos analizando. Consiste en medir un coeficiente de correlación entre pares de variables (X, Y), 
        para poder medir el grado de similitud (que tan dependientes son) e ir discriminando variables. ''')
        st.write(df.corr())

        st.subheader("Top 10 valores")
        val = st.selectbox('variable a analizar', options=numeric_columns, key="3")
        st.write(df.corr()[val].sort_values(ascending=False)[:10], '\n')
        
        fig2 = plt.figure()
        st.header("Evaluación Visual")
        st.caption('''Es importante hacer una evaluación visual de los datos a través de gráficos de dispersión,
        hacer un plot (graficar) entre pares de variables que posiblemente estén correlacionados. Esto nos ayuda a 
        detectar si hay cierta linealidad en la data, y saber si hay una fuerte o débil correlación.''')
        x_val = st.selectbox('X', options=numeric_columns, key="2")
        y_val = st.selectbox('Y', options=numeric_columns, key="2")
        sns.scatterplot(data=df , x = x_val , y = y_val)
        st.pyplot(fig2)

        st.header("Mapa de calor")
        st.caption('''También es posible ver esta matriz en un mapa de calor, donde la intensidad del color es
        proporcional a los valores de los coeficientes de correlación.''')
        fig3 = plt.figure()
        sns.set(font_scale=0.5)
        sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, annot_kws={"size": 6})
        st.pyplot(fig3)

        st.subheader("Lectura inferior del mapa de calor")
        fig4 = plt.figure()
        sns.set(font_scale=0.5)
        MatrizInf = np.triu(df.corr())
        sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf, annot_kws={"size": 6})
        st.pyplot(fig4)

        st.subheader("Lectura superior del mapa de calor")
        fig5 = plt.figure()
        sns.set(font_scale=0.5)
        MatrizSup = np.tril(df.corr())
        sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, mask=MatrizSup, annot_kws={"size": 6})
        st.pyplot(fig5)

        st.header("Elección de variables")
        st.caption('''Observando la matriz de correlaciones podemos detectar las dependencia altas e ir discriminando, 
        para quedarnos con las variables más significativas. Todo esto depende del algoritmo a utilizar y una vez identificadas, 
        las variables a eliminar, procedemos a quitarlas de nuestro dataframe. Sugerimos eliminar las variables no numericas\n
        Ingresa las variables a eliminar separadas con un espacio, posterior presiona Eliminar variable''')
        df.to_csv('data/main_data_new.csv', index=False)
        df_new = pd.read_csv('data/main_data_new.csv')
        st.subheader("Eliminar variables del dataframe")
        text_input = st.text_input("Ingresa las variables a eliminar del dataframe", "")
        lista = text_input.split()
        if st.button("Eliminar variables"):
            df_new.drop(lista, axis=1, inplace=True)
            df_new.to_csv('data/main_data_final.csv', index=False)
            st.write(df_new)


