import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.pyplot import figure
import seaborn as sns  
import os
import plotly.express as px
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, max_error, r2_score
from sklearn import model_selection

def app():
    st.title("Pronóstico con Regresión Lineal Múltiple ")
    st.caption('''La regresión es una forma estadística de establecer una relación entre una variable dependiente y 
    un conjunto de variables independientes. Es un enfoque muy simple para el aprendizaje supervisado, su propósito es 
    establecer un modelo para la relación entre un cierto número de características y una variable objetivo continua, 
    calcula una ecuación que minimiza la distancia entre la línea ajustada (recta) y todos los puntos de datos.''')
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data.csv')
        st.header("Dataframe a trabajar")
        st.write(df)
        st.subheader("Seleccione las variables que quiere graficar")
        text_input = st.text_input("Ingresa la variable 1")
        text_input2 = st.text_input("Ingresa la variable 2")
        try:
            fig1 = plt.figure()
            fig1.set_figwidth(25)
            fig1.set_figheight(12)
            plt.xlabel(text_input)
            plt.ylabel(text_input2)
            #plt.figure(figsize=(1, 1))
            plt.plot(df[text_input], df[text_input2], color='green', marker='o', label=text_input2)
            plt.grid(True)
            plt.legend()
            st.pyplot(fig1)
        except Exception as e:
            st.error("Para que exista una gráfica ingrese las variables a graficar en la parte de arriba")
        
        st.header("Seleccionar variables predictoras y la variable a pronosticar")
        st.caption('''Recuerda que antes de realizar este paso ya debes tener tus variables más significativas seleccionadas,
        esto para tener un mejor resultado. Si aun no lo has hecho regresa al apartado de "Selección de Caracteristicas" y
        luego regresa''')
        text_input3 = st.text_input("Ingresa las variables predictoras separadas por un espacio", "")
        lista = text_input3.split()
        X = np.array(df[lista])
        st.subheader("Matriz de las variables predictoras")
        st.write(pd.DataFrame(X))
        text_input4 = st.text_input("Ingresa la variable a pronosticar", "")
        lista2 = text_input4.split()
        Y = np.array(df[lista2])
        st.subheader("Matriz de la variable a pronosticar")
        st.write(pd.DataFrame(Y))

        st.header("División de datos")
        st.caption('''Debido a que la información de las variables es demasiada se procede a separar/dividir los datos. 
        Se utiliza un porcentaje de los datos para entrenamiento y el resto para pruebas. Eliga el porcentaje de su agrado
        para entrenamiento. Ejemplo: Si quiero 20% ingreso el valor 20, sin el signo %''')
        number = st.number_input("Ingresa el porcentaje %: ", min_value=0, max_value=100, key=1)
        porcentaje = number/100
        try:
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = porcentaje, random_state = 1234, shuffle = True)
        except Exception as e:
            st.error("Ingrese un porcentaje válido")
        try:
            st.subheader("Matriz con las variables predictoras (Datos divididos)")
            st.write(pd.DataFrame(X_train))
            st.subheader("Matriz con las variable a pronosticar (Datos divididos)")
            st.write(pd.DataFrame(Y_train))
        except Exception as e:
            st.error("Se necesita el porcentaje para poder visualizar las matrices")
        try: 
            st.markdown('''Procedemos a entrenar al modelo y generar un pronóstico con los datos de prueba
            que a continuación podra observar.''')
            RLMultiple = linear_model.LinearRegression()
            RLMultiple.fit(X_train, Y_train)
            Y_Pronostico = RLMultiple.predict(X_test)
            st.write(pd.DataFrame(Y_Pronostico))
            st.markdown('''Teniendo el pronóstico se calcula el Score (comparación del pronóstico contra en valor real).
            Este valor nos ayudará a identificar el porcentaje de efectividad con el que nuestro modelo esta pronosticando.''')
            st.write(r2_score(Y_test, Y_Pronostico))
            score = (r2_score(Y_test, Y_Pronostico))*100
            st.write('Lo que nos indica que el pronostico se logrará con un ',score,'%')
        except Exception as e:
            st.error("Se necesitan los datos entrenados para visualizar esta parte")
            
        st.header("Obtención de los coeficientes, intercepto, error y score")
        try:
            st.write('Coeficientes: \n', RLMultiple.coef_)
            st.write('Intercepto: \n', RLMultiple.intercept_)
            st.write("Residuo: %.4f" % max_error(Y_test, Y_Pronostico))
            st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
            st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))
            st.write('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))
        except Exception as e:
            st.error("Se necesitan los datos entrenados para visualizar esta parte")
            
