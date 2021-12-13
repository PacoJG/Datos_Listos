import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import io 
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def app():
    st.title("Análisis de Componentes Principales (PCA)")
    st.caption('''El Análisis de Componentes Principales es uno de los métodos estadísticos de extracción de datos 
    más populares, nos ayuda a perder la menor cantidad de información (varianza) posible. Cuando contamos con un gran 
    número de variables ACP nos permite reducirlas a un número menor de variables transformadas (componentes principales) 
    que expliquen gran parte de la variabilidad en los datos, esto gracias al trabajo a nivel matricial en donde obtenemos 
    las variables con una carga más significativas que otras (varianza).''')
    if 'main_data_final.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data_final.csv')
        st.header("Estandarización de los datos")
        st.caption('''Pasar a una escala numérica simultánea en todas las variables, se normalizan para que todas 
        las variables tengan el mismo impacto en el análisis. Se hace para que las variables de mayor impacto 
        (magnitudes) no condicionen el análisis y así todas estén en un mismo rango.''')
        estandarizar = StandardScaler()
        estandarizar.fit(df)
        MEstandarizada = estandarizar.transform(df) 
        st.write(pd.DataFrame(MEstandarizada, columns=df.columns))

        st.header("Decidir el número de componentes principales")
        st.caption('''La idea es reducir la dimensionalidad de la data, es por ello que se elige el 90% de relevancia 
        (varianza acumulada) para tener un 10% de margen para poder manipular y eliminar algunas variables que no cumplan 
        la condición de carga. Y no menos de 75% para no sacrificar la información, es decir entre el 75% y 90% de varianza total. ''')
        var_pca = PCA(n_components=None)
        var_pca.fit(MEstandarizada)
        X_pca = var_pca.transform(MEstandarizada)
        varianza = var_pca.explained_variance_ratio_
        st.subheader("Elige el porcentaje de relevancia")
        num_var = st.slider('Un número entre 0 y 20', min_value=0, max_value=20)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Varianza acumulada:",sum(varianza[0:num_var]))
            sum_varianza = str(sum(varianza[0:num_var]))
            st.write("Con " + str(num_var) + " componentes se tiene el " + sum_varianza +"% de varianza acumulada" )
        with col2:
            eigenvalues = var_pca.explained_variance_
            st.write("Eigenvalues:",eigenvalues)

        st.header("Gráfica de la varianza")
        st.caption('''Se identifica mediante una gráfica el grupo de componentes con mayor varianza.''')
        fig1 = plt.figure()
        plt.plot(np.cumsum(var_pca.explained_variance_ratio_))
        plt.xlabel('Número de componentes')
        plt.ylabel('Varianza acumulada')
        plt.grid()
        st.pyplot(fig1)

        st.header("Proporción de relevancias")
        st.caption('''La importancia de cada variable se refleja en la magnitud de los valores en los componentes
        (mayor magnitud es sinónimo de mayor importancia). Con ayuda del anterior paso podemos concluir que se va a 
        trabajar con un máximo de ''' + str(num_var) +''' componentes, que quiere decir que dentro de dichos componentes 
        hay un grupo de variables con el cual se trabajará ya que trae información significativa. Y en cada uno de estos 
        componentes se tienen que elegir las cargas máximas. Por obvias razones se decartan los demas componentes''')
        st.caption("Se revisa cada variable usted elige con que porcentaje de carga mayor trabajará y asi eliminar las demas")
        componentes = pd.DataFrame(abs(var_pca.components_), columns=df.columns)
        st.write(componentes)
        df_new = df

        st.header("Eliminar variables del dataframe")
        st.caption('''Para finalizar el Análisis de componentes elimina las variables que decidiste para quedarte con 
        las más significativas''')
        text_input = st.text_input("Ingresa las variables a eliminar del dataframe")
        lista = text_input.split()
        if st.button("Eliminar variable"):
            df_new.drop(lista, axis=1, inplace=True)
            df_new.to_csv('data/main_data_pca.csv', index=False)
            st.write(df_new)
        if st.button("Seguir con el mismo dataframe"):
            df_new.to_csv('data/main_data_pca.csv', index=False)
            st.write(df_new)






