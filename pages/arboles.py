import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.pyplot import figure
import seaborn as sns  
import os
import plotly.express as px
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

def app():
    st.title("Árboles de decisión")
    st.caption('''Los árboles de decisión es uno de los algoritmos más utilizados en el aprendizaje automático ya que 
    proveen de una herramienta de clasificación muy potente. Su uso en el manejo de datos la hace ganar en popularidad 
    dadas las posibilidades que brinda y la facilidad con que son comprendidos sus resultados por cualquier usuario.''')
    st.caption('''Son algoritmos estadísticos o técnicas de machine learning que nos permiten la construcción de modelos 
    predictivos de analítica de datos basados en su clasificación según ciertas características o propiedades, o en la 
    regresión mediante la relación entre distintas variables para predecir el valor de otra.\n
    Estructura: \n
    •Nodo principal. Representa toda la población (todos los datos) que posteriormente se dividirá. También cuenta como un nodo de decisión.\n
    •Nodo de decisión. Se encarga de dividir los datos, dependiendo de una decisión. Se toman dos caminos.\n
    •Nodo hoja. Es donde recae la decisión final.\n
    •Profundidad. La profundidad indica los niveles que tiene el árbol de decisión.''')
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

        st.subheader("Nota: Sobreajuste")
        st.caption('''¿Qué es el sobreajuste? Es cuando el modelo explica todo el conjunto de datos
        de entrenamiento, es decir, cubre el 100% de los posibles casos sobre el dataframe. Y lo que queremos es encontrar
        patrones que generalizan, es por ello que Datos Listos evita el sobreajuste. En cada tipo de árbol encontraras
        varias opciones que podras modificar\n.''' )
        st.header("Elije que tipo de árbol quieres visualizar")

        
        with st.expander("Árbol de decisión (Regresión/Pronóstico)"):
            st.header("Árbol de decisión (Regresión/Pronóstico) ")
            st.caption('''En los modelos de regresión (pronóstico) se intenta predecir el valor de una variable en función de 
            otras variables que son independientes entre sí. Por ejemplo, queremos predecir el precio de venta del terreno en 
            función de variables como su localización, superficie, distancia a la playa, etc. Cuando usamos regresión, el resultado 
            será un valor numérico y el posible resultado no forma parte de un conjunto predefinido, sino que puede tomar cualquier 
            posible valor.''')

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
            •max_depth. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el sobreajuste, pero también puede provocar underfitting.\n
            •min_samples_leaf. Indica la cantidad mínima de datos que debe tener un nodo hoja.\n
            •min_samples_split. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.\n
            •criterion. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE).''')

            try:
                var1 = st.slider('Ingrese profundidad del arbol', min_value=1, max_value=50)
                var2 = st.slider('Ingrese el valor de la variable min_samples_split', min_value=1, max_value=50)
                var3 = st.slider('Ingrese el valor de la variable min_samples_leaf', min_value=1, max_value=50)
                PronosticoAD = DecisionTreeRegressor(max_depth=var1, min_samples_split=var2, min_samples_leaf=var3)
                PronosticoAD.fit(X_train, Y_train)
                col1, col2 = st.columns([1,3])
                with col1:
                    st.markdown("Pronóstico")
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    st.write(pd.DataFrame(Y_Pronostico))
                with col2:
                    st.markdown("valores pronosticados vs valores reales.")
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.write(Valores)
                    st.caption("Si los valores no son tan diferentes el resultado puede clasificarse como eficiente")
            except Exception as e:
                st.error('''Se necesita entrenar al modelo para mostrar el pronostico. Recuerda que todas las variables
                tanto preictoras como la variavble a predecir, deben ser númericas para este tipo de árbol''')
            
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
                    st.write('Criterio: \n', PronosticoAD.criterion)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))
                    st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
                with col4:
                    Importancia = pd.DataFrame({'Variable': list(df[lista]),'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.write('Importancia variables: \n',Importancia)
            except Exception as e:
                st.error("Debe de existir valores pronosticado para mostrar los parámetros")
            try:
                st.header("Impresión del árbol")
                fig2 = plt.figure()
                fig2.set_figwidth(30)
                fig2.set_figheight(15)
                plot_tree(PronosticoAD, feature_names = lista)
                st.pyplot(fig2)
                st.header("Reglas del árbol")
                Reporte = export_text(PronosticoAD, feature_names = lista)
                st.code(Reporte, language='python')
            except Exception as e:
                st.error("No existe árbol que imprimir")

            

        with st.expander("Árbol de decisión (Clasificación)"):
            st.header("Árbol de decisión (Clasificación) ")
            st.caption('''En los modelos de clasificación queremos predecir el valor de una variable mediante la 
            clasificación de la información en función de otras variables (tipo, pertenencia a un grupo...). Por ejemplo, 
            queremos pronosticar qué personas comprarán un determinado producto, clasificando entre clientes y no clientes, 
            o qué marcas de portátiles comprará cada persona mediante la clasificación entre las distintas marcas. Los valores 
            a predecir son predefinidos, es decir, los resultados están definidos en un conjunto de posibles valores.''')

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
            •max_depth. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el sobreajuste, pero también puede provocar underfitting.\n
            •min_samples_leaf. Indica la cantidad mínima de datos que debe tener un nodo hoja.\n
            •min_samples_split. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.\n
            •criterion. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE).''')
            
            try:
                var4 = st.slider('Ingrese profundidad del arbol', min_value=1, max_value=50, key=2)
                var5 = st.slider('Ingrese el valor de la variable min_samples_split', min_value=1, max_value=50, key=2)
                var6 = st.slider('Ingrese el valor de la variable min_samples_leaf', min_value=1, max_value=50, key=2)
                ClasificacionAD = DecisionTreeClassifier(max_depth=var4, min_samples_split=var5, min_samples_leaf=var6)
                ClasificacionAD.fit(X2_train, Y2_train)
                col4, col5 = st.columns([1,3])
                with col4:
                    st.markdown("Clasificaciones")
                    Y_Clasificacion = ClasificacionAD.predict(X_validation)
                    st.write(pd.DataFrame(Y_Clasificacion))
                with col5:
                    st.markdown("valores pronosticados vs valores reales.")
                    Valores2 = pd.DataFrame(Y_validation, Y_Clasificacion)
                    st.write(Valores2)
                st.caption("Si los valores no son tan diferentes, el resultado puede clasificarse como eficiente")
            except Exception as e:
                st.error("Se necesita entrenar al modelo para mostrar el pronostico")
            
            try:
                score2 = ClasificacionAD.score(X_validation, Y_validation)
                st.write('Exactitud promedio de la validación ',score2)
                st.subheader("Matriz de clasificación")
                Y_Clasificacion = ClasificacionAD.predict(X_validation)
                Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación']) 
                st.write(Matriz_Clasificacion)
            except Exception as e:
                st.error("Se necesita entrenar al modelo para mostrar la matriz de clasificación")
            
            try:
                st.header("Obtención de los parámetros del modelo")
                col6, col7 = st.columns(2)
                with col6:
                    st.write('Criterio: \n', ClasificacionAD.criterion)
                    st.write("Exactitud: \n", ClasificacionAD.score(X_validation, Y_validation))
                    st.write(classification_report(Y_validation, Y_Clasificacion))
                with col7:
                    Importancia2 = pd.DataFrame({'Variable': list(df[lista]),'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.write('Importancia variables: \n',Importancia2)
            except Exception as e:
                st.error("Debe de existir valores pronosticado para mostrar los parámetros")
            
            try:
                st.header("Impresión del árbol")
                fig3 = plt.figure()
                fig3.set_figwidth(30)
                fig3.set_figheight(15)
                plot_tree(ClasificacionAD, feature_names = lista, class_names = Y_Clasificacion)
                st.pyplot(fig3)
                st.header("Reglas del árbol")
                Reporte2 = export_text(ClasificacionAD, feature_names = lista)
                st.code(Reporte2, language='python')
            except Exception as e:
                st.error("No existe árbol que imprimir")