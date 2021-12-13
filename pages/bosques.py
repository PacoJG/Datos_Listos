import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.pyplot import figure
import seaborn as sns  
import os
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def app():
    st.title("Bosques Aleatorios")
    st.caption('''Los Árboles de decisión tienen una gran desventaja: el overfitting. Esto quiere decir que 
    en general funcionan muy bien durante el entrenamiento, pero no tanto cuando introducimos datos nuevos 
    (es decir cuándo queremos hacer predicciones). Y es esto dio origen a los Bosques Aleatorios (Random Forests), 
    uno de los algoritmos más poderosos y más usados del Machine Learning. ''')
    st.caption('''En lugar de entrenar un único árbol se entrenan varios, usualmente decenas o cientos. 
    Es importante mencionar que cada árbol del bosque se entrena con una parte distinta del dataset original. 
    Y acá es dónde viene una característica importante: este subset de entrenamiento se selecciona aleatoriamente. 
    De ahí precisamente el nombre “Bosques Aleatorios”. ''')
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data.csv')
        st.header("Dataframe original")
        st.write(df)
        st.header("Seleccionar variables predictoras y la variable a pronosticar")
        st.caption('''Recuerda que antes de realizar este paso ya debes tener tus variables más significativas seleccionadas,
        esto para tener un mejor resultado. Si aun no lo has hecho regresa al apartado de "Selección de Caracteristicas" y
        luego regresa''')
        text_input1 = st.text_input("Ingresa las variables predictoras separadas por un espacio", "")
        lista = text_input1.split()
        X = np.array(df[lista])
        st.subheader("Matriz de las variables predictoras")
        st.write(pd.DataFrame(X))
        text_input2 = st.text_input("Ingresa la variable a pronosticar", "")
        lista2 = text_input2.split()
        Y = np.array(df[lista2])
        st.subheader("Matriz de la variable a pronosticar")
        st.write(pd.DataFrame(Y))

        with st.expander("Bosques Aleatorios (Pronóstico)"):
            st.header("Bosques Aleatorios (Pronóstico)")
            st.caption('''Con el Bosque Aleatorio ya entrenado es fácil realizar la predicción.se realiza la regresión
            individual y el valor final sería simplemente el promedio de la predicción hecha por cada árbol.''')
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
            
            st.markdown('''Procedemos a entrenar el arbol, recuerda que puedes modificar los parametros de este entrenamiento \n
            •n_estimators. Número de árboles a entrenar\n
            •max_depth. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el sobreajuste, pero también puede provocar underfitting.\n
            •min_samples_leaf. Indica la cantidad mínima de datos que debe tener un nodo hoja.\n
            •min_samples_split. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.''')
            try:
                var1 = st.slider('Ingrese el número de árboles a entrenar', min_value=1, max_value=500, key=1)
                var2 = st.slider('Ingrese profundidad del arbol', min_value=1, max_value=50, key=1)
                var3 = st.slider('Ingrese el valor de la variable min_samples_split', min_value=1, max_value=50, key=1)
                var4 = st.slider('Ingrese el valor de la variable min_samples_leaf', min_value=1, max_value=50, key=1)
                PronosticoBA = RandomForestRegressor(n_estimators=var1, max_depth=var2, min_samples_split=var3, min_samples_leaf=var4, random_state=0)
                PronosticoBA.fit(X_train, Y_train)
                col1, col2 = st.columns([1,3])
                with col1:
                    st.markdown("Pronóstico")
                    Y_Pronostico = PronosticoBA.predict(X_test)
                    st.write(pd.DataFrame(Y_Pronostico))
                with col2:
                    st.markdown("valores pronosticados vs valores reales.")
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.write(Valores)
                    st.caption("Si los valores no son tan diferentes el resultado puede clasificarse como eficiente")
            except Exception as e:
                st.error('''Se necesita entrenar al modelo para mostrar el pronostico. Recuerda que todas las variables
                tanto predictoras como la variavble a predecir, deben ser númericas para este tipo de bosque aleatorio''')
        
            try:
                st.subheader("Gráfica de los valores pronosticados vs los valores reales")
                fig1 = plt.figure()
                fig1.set_figwidth(25)
                fig1.set_figheight(12)
                plt.plot(Y_test, color='green', marker='o', label='Y_test')
                plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                plt.grid(True)
                plt.legend()
                st.pyplot(fig1)
                score = (r2_score(Y_test, Y_Pronostico))*100
                st.write('Obtenemos un score de ',score,'%')
            except Exception as e:
                st.error("Debe de existir valores pronosticado para mostrar la gráfica")
            
            try:
                st.header("Obtención de los parámetros del modelo")
                col3, col4 = st.columns(2)
                with col3:
                    st.write('Criterio: \n', PronosticoBA.criterion)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))
                    st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
                with col4:
                    Importancia = pd.DataFrame({'Variable': list(df[lista]),'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.write('Importancia variables: \n',Importancia)
            except Exception as e:
                st.error("Debe de existir valores pronosticado para mostrar los parámetros")
            
            try:
                st.header("Estimadores")
                est = st.number_input("Ingresa el número de estimadores: ", min_value=0, max_value=100, key=2)
                Estimador = PronosticoBA.estimators_[est]
                st.write(Estimador)
                st.header("Impresión del árbol dado los estimadores")
                fig2 = plt.figure()
                fig2.set_figwidth(30)
                fig2.set_figheight(15)
                plot_tree(Estimador, feature_names = lista)
                st.pyplot(fig2)
                st.header("Reglas del árbol dado los estimadores")
                Reporte1 = export_text(Estimador, feature_names = lista)
                st.code(Reporte1, language='python')
            except Exception as e:
                st.error("No existe árbol que imprimir")
        
        with st.expander("Bosque Aleatorio (Clasificación)"):
            st.header("Árbol de decisión (Clasificación) ")
            st.caption('''Con el Bosque Aleatorio ya entrenado es fácil realizar la predicción. Simplemente se 
            introduce el nuevo dato a cada árbol, se realiza la clasificación individual y se escoge la categoría 
            asignada por la mayoría de los árboles.''')

            st.header("División de datos")
            st.caption('''Debido a que la información de las variables es demasiada se procede a separar/dividir los datos. 
            Se utiliza un porcentaje de los datos para entrenamiento y el resto para pruebas. Eliga el porcentaje de su agrado
            para entrenamiento. Ejemplo: Si quiero 20% ingreso el valor 20, sin el signo %''')
            number2 = st.number_input("Ingresa el porcentaje %: ", min_value=0, max_value=100, key=2)
            porcentaje2 = number2/100
            try:
                X2_train, X_validation, Y2_train, Y_validation = model_selection.train_test_split(X, Y, test_size = porcentaje2, random_state = 0, shuffle = True)
            except Exception as e:
                st.error("Ingrese un porcentaje válido")
            try:
                st.subheader("Matriz con las variables predictoras (Datos divididos)")
                st.write(pd.DataFrame(X2_train))
                st.subheader("Matriz con las variable a pronosticar (Datos divididos)")
                st.write(pd.DataFrame(Y2_train))
            except Exception as e:
                st.error("Se necesita el porcentaje para poder visualizar las matrices")
            
            st.markdown('''Procedemos a entrenar el arbol, recuerda que puedes modificar los parametros de este entrenamiento \n
            •n_estimators. Número de árboles a entrenar\n
            •max_depth. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el sobreajuste, pero también puede provocar underfitting.\n
            •min_samples_leaf. Indica la cantidad mínima de datos que debe tener un nodo hoja.\n
            •min_samples_split. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.''')
            try:
                var5 = st.slider('Ingrese el número de árboles a entrenar', min_value=1, max_value=500, key=2)
                var6 = st.slider('Ingrese profundidad del arbol', min_value=1, max_value=50,key=2)
                var7 = st.slider('Ingrese el valor de la variable min_samples_split', min_value=1, max_value=50, key=2)
                var8 = st.slider('Ingrese el valor de la variable min_samples_leaf', min_value=1, max_value=50, key=2)
                ClasificacionBA = RandomForestClassifier(n_estimators=var5, max_depth=var6, min_samples_split=var7, min_samples_leaf=var8, random_state=0)
                ClasificacionBA.fit(X2_train, Y2_train)
                col4, col5 = st.columns([1,3])
                with col4:
                    st.markdown("Clasificaciones")
                    Y_Clasificacion = ClasificacionBA.predict(X_validation)
                    st.write(pd.DataFrame(Y_Clasificacion))
                with col5:
                    st.markdown("valores pronosticados vs valores reales.")
                    Valores2 = pd.DataFrame(Y_validation, Y_Clasificacion)
                    st.write(Valores2)
                st.caption("Si los valores no son tan diferentes, el resultado puede clasificarse como eficiente")
            except Exception as e:
                st.error("Se necesita entrenar al modelo para mostrar el pronostico")
            
            try:
                score2 = ClasificacionBA.score(X_validation, Y_validation)
                st.write('Exactitud promedio de la validación ',score2)
                st.subheader("Matriz de clasificación")
                Y_Clasificacion = ClasificacionBA.predict(X_validation)
                Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación']) 
                st.write(Matriz_Clasificacion)
            except Exception as e:
                st.error("Se necesita entrenar al modelo para mostrar la matriz de clasificación")
            
            try:
                st.header("Obtención de los parámetros del modelo")
                col6, col7 = st.columns(2)
                with col6:
                    st.write('Criterio: \n', ClasificacionBA.criterion)
                    st.write("Exactitud: \n", ClasificacionBA.score(X_validation, Y_validation))
                    st.write(classification_report(Y_validation, Y_Clasificacion))
                with col7:
                    Importancia2 = pd.DataFrame({'Variable': list(df[lista]),'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.write('Importancia variables: \n',Importancia2)
            except Exception as e:
                st.error("Debe de existir valores pronosticado para mostrar los parámetros")
            
            try:
                st.header("Estimadores")
                est2 = st.number_input("Ingresa el número de estimadores: ", min_value=0, max_value=100, key=3)
                Estimador2 = ClasificacionBA.estimators_[est2]
                st.write(Estimador2)
                st.header("Impresión del árbol dado los estimadores")
                fig3 = plt.figure()
                fig3.set_figwidth(30)
                fig3.set_figheight(15)
                plot_tree(Estimador2, feature_names = lista, class_names = Y_Clasificacion)
                st.pyplot(fig3)
                st.header("Reglas del árbol dado los estimadores")
                Reporte2 = export_text(Estimador2, feature_names = lista)
                st.code(Reporte2, language='python')
            except Exception as e:
                st.error("No existe árbol que imprimir")