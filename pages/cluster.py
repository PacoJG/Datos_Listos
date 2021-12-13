import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering  
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

def app():
    st.title("Clusterización")
    st.caption('''Un problema de clasificación no supervisada parte de un conjunto de objetos cada uno de los cuales 
    está caracterizado por varias variables, y a partir de dicha información trata de obtener grupos de objetos, de tal 
    manera que los objetos que pertenecen a un grupo sean muy homogéneos entre sí y por otra parte la heterogeneidad 
    entre los distintos grupos sea muy elevada. Expresado en términos de variabilidad hablaríamos de minimizar la variabilidad 
    dentro de los grupos para al mismo tiempo maximizar la variabilidad entre los distintos grupos.''')
    if 'main_data_pca.csv' not in os.listdir('data'):
        st.markdown("Porfavor sube un archivo en la pestaña `Cargar Datos`")
    else:
        df = pd.read_csv('data/main_data_pca.csv')
        st.header("Dataframe a trabajar")
        st.caption('''El dataframe se obtuvo del apartado de "Análisis de Componentes Principales",
        es  necesario realizar dicho apartado para poder continuar ''')
        st.write(df)
        MatrizVaraibles = df
        df_2 = df
        st.header("Estandarización de los datos")
        estandarizar = StandardScaler()                             
        MEstandarizada = estandarizar.fit_transform(MatrizVaraibles)
        st.write(pd.DataFrame(MEstandarizada, columns=df.columns))

        with st.expander("Clustering Jerárquico"):
            st.header("Clustering Jerárquico")
            st.caption('''El clustering jerárquico es un método de minería de datos para agrupar, llamados clústers. 
            El algoritmo de clúster jerárquico agrupa los datos basándose en la distancia entre cada uno y buscando 
            que los datos que están dentro de un clúster sean los más similares entre sí. En una representación gráfica, 
            el algoritmo organiza los elementos, en una estructura en forma de árbol.''')
            fig1 = plt.figure()
            plt.xlabel('Observaciones')
            plt.ylabel('Distancia')
            shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean')) 
            st.pyplot(fig1)
            st.caption('''Los colores representan el número de clústers asignados. Para poder observar que variable 
            pertenece a cada clúster se realiza una matriz de etiquetado. Para ello, a continuación ingresa el número de clusters''')
            num_cluster = st.number_input("Ingresa el número de clusters", min_value=1, max_value=1000, key=1)
            MJerarquico = AgglomerativeClustering(n_clusters=num_cluster, linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MEstandarizada)
            st.write(MJerarquico.labels_)
            df['clusterH'] = MJerarquico.labels_
            st.header("Matriz con la etiqueta del cluster")
            st.write(df)
            st.subheader("Cantidad de elementos en los clusters")
            st.write(df.groupby(['clusterH'])['clusterH'].count())
            st.caption('''Si usted lo desea puede visualizar todos los elementos de un sólo cluster.
            A continuación eliga que cluster quiere visualizar''')
            cluster_num = st.number_input("Ingresa el número de cluster", min_value=1, max_value=1000, key=2)
            st.write(df[df.clusterH == cluster_num])
        
            st.caption('''Se obtiene la media de todas las variables dentro de cada cluster (centroides),
            esto para recolectar la información necesaria y darle herramientas para que sus conclusiones.''')
            centroidesH = df.groupby(['clusterH']).mean()
            st.write(centroidesH)

        with st.expander("Clustering Particional"):
            st.header("Clustering Particional")
            st.caption('''En el clustering particional el objetivo es obtener una partición de los objetos en conjuntos o
            clusters de datos de tal forma que todos los objetos pertenezcan a alguno de los k clusters
            posibles y que por otra parte los clusters sean disjuntos, el algoritmo más usado es K-means. K-means es un algoritmo de 
            aprendizaje no supervisada (los datos no tienen etiquetas) que agrupa objetos en k grupos basándose en sus 
            características, de modo que los miembros de un grupo sean similares. El agrupamiento se realiza minimizando 
            (optimizar) la suma de distancias entre cada objeto al centroide de un clúster. ''')
            SSE = []
            for i in range(2, 12):
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(MEstandarizada)
                SSE.append(km.inertia_)
    
            st.caption('''Se usará el algoritmo jerárquico particional, este trabaja con distancias euclidianas. 
            Posterior a esto, se aplica el método del codo para hacer una aproximación de cuántos clústers se ocuparán.
            En ocasiones esta codo agudo no se podrá visualizar en la gráfica, es por ello que nosotros le facilitamos el número exacto.''')
            fig2 = plt.figure()
            plt.plot(range(2, 12), SSE, marker='o')
            plt.xlabel('Cantidad de clusters *k*')
            plt.ylabel('SSE')
            plt.title('Elbow Method')
            st.pyplot(fig2)

            kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
            #st.markdown("El número de clústers que necesita es:")
            MPart_num = kl.elbow 
            st.write('El número de clústers que necesita es: ',kl.elbow)
            st.write('\n')

            st.subheader("Se crearon las etiquetas de los elementos en los clusters")
            MParticional = KMeans(n_clusters=MPart_num, random_state=0).fit(MEstandarizada)
            MParticional.predict(MEstandarizada)
            st.write(MParticional.labels_)
            df['clusterP'] = MParticional.labels_
            st.header("Matriz con la etiqueta del cluster")
            st.write(df)
            st.subheader("Cantidad de elementos en los clusters")
            st.write(df.groupby(['clusterP'])['clusterP'].count())
            st.caption('''Si usted lo desea puede visualizar todos los elementos de un sólo cluster.
            A continuación eliga que cluster quiere visualizar''')
            cluster_num_2 = st.number_input("Ingresa el número de cluster", min_value=1, max_value=1000, key=3)
            st.write(df[df.clusterP == cluster_num_2])
            st.caption('''Se obtiene la media de todas las variables dentro de cada cluster (centroides),
            esto para recolectar la información necesaria y darle herramientas para que sus conclusiones.''')
            centroidesP = df.groupby(['clusterP']).mean()
            st.write(centroidesP)



