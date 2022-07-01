



#Librerías necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pearsonr
import scipy
import random


from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model



import statsmodels.api as sm
import pingouin as pg
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import bernoulli

plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import stats

from seaborn import load_dataset

salir1 = False #bandera
opcion1 = 0
opcion2 = 0

df = pd.read_csv('gapminder2.csv', index_col=0)
df1 = pd.read_csv('Experimento_a.csv', index_col=0)
df2 = pd.read_csv('Experimento_b.csv', index_col=0)

experimento_a = [0] * 10
experimento_b = [0] * 10
for i in range(10):
    experimento_a[i] = random.random()
    experimento_b[i] = random.random()

while not salir1:
    salir2 = False
    print("        TALLER 3           ")
    print("1. Opción 1: Punto 1")
    print("2. Opción 2: Punto 2")
    print("3. Opción 3: Punto 3")
    print("4. Opción 4: Salir")
    print()
    opcion1 = int(input("Ingrese una opción: "))
    print()
    print()
    if opcion1 == 1:
        while not salir2:
            print("1. Exportar el conjunto de datos gapminder en formato “xlsx”. El 10 % de los valores de las columnas lifeExp, pop, y gdpPercap serán reemplazados de forma aleatoria por valores no asignados NA")
            print("2. Importar el archivo gapminder en formato “xlsx”")
            print("3. Graficar el diagrama de dispersión lifeExp vs pop")
            print("4. Graficar el diagrama de dispersión gdpPercap vs pop")
            print("5. Graficar los diagramas de cajas de la variable gdpPercap discriminados por continentes desde 1990 a 2007")
            print("6. Salir")
            print()
            opcion2 = int(input("Ingrese una opción: "))
            if opcion2 == 1:
                print()
                print()
                df = pd.read_csv('gapminder2.csv', index_col=0)
                dfnan = df.copy()
                for colname in ['lifeExp', 'pop', 'gdpPercap']:
                    indices = np.random.permutation(len(dfnan))
                    indices = indices[:100]
                    dfnan[colname].loc[indices] = np.nan
                    dataFramenew = dfnan


                print(dataFramenew)
                dataFramenew.to_csv('ej.csv')
                print(type(dfnan))

            elif opcion2 == 2:
                print()
                print()
                df = pd.read_csv('gapminder2.csv', index_col=0)
                print(df)
                print()
            elif opcion2 == 3:
                print()
                lifeexp = df.iloc[:, 3]  # Primera columna
                pop = df.iloc[:,4]
                fig, ax = plt.subplots()
                ax.scatter(lifeexp,pop)
                plt.show()


                #plt.scatter()
                print()
                print()
            elif opcion2 == 4:
                print()
                print()
                gdcap = df.iloc[:, 5]  # Primera columna
                pop = df.iloc[:, 4]
                fig, ax = plt.subplots()
                ax.scatter(gdcap, pop)
                plt.show()
                print()
            elif opcion2 == 5:
                print()
                print()
                print("opción 5")
                print()
            elif opcion2 == 6:
                salir2 = True
            else:
                print("Digíte una opción válida")

    elif opcion1 == 2:
        while not salir2:
            print()
            print("1. Cargar dos archivos de datos en formato “csv” llamados “Experimento a.csv” y “Experimento b.csv” e indicar si la diferencia en la media de los datos es estadísticamente significativa.")
            print("2. Cargar dos archivos de datos en formato “csv” llamados “Experimento a.csv” y “Experimento b.csv” y mostrar en pantalla la correlación de Pearson y Spearman de los datos.")
            print("3. Cargar dos archivos de datos en formato “csv” llamados“Experimento a.csv” y “Experimento b.csv” y graficar el diagrama de dispersión y la línea recta que aproxime los datos calculada por una regresión lineal por mínimos cuadrados.")
            print("4. Salir")
            print()
            opcion2 = int(input("Ingrese una opción"))
            if opcion2 == 1:
                print()
                print()


                print(experimento_a)
                print(experimento_b)
                r, p = spearmanr(experimento_a, experimento_b)

                if p > 0.05:
                    print()
                    print("La diferencia en la media de los datos no es estadísticamente significativa")
                    print()
                else:
                    print()
                    print("La diferencia en la media de los datos es estadísticamente significativa")
                    print()






                print()
            elif opcion2 == 2:
                print()

                pears = pearsonr(experimento_a, experimento_b)
                print("La correlación de Pearson es ", pears,"\n")
                sper = spearmanr(experimento_a, experimento_b)
                print("La correlación de Spearman es: ", sper)
            elif opcion2 == 3:
                print()
                matriz_a = np.array(experimento_a)
                matriz_b = np.array(experimento_b)
                z = np.array([7, 8, 9, 10, 11, 12])  # Periodos que deseo pronosticar

                # Graficamos los datos de entrada
                plt.scatter(matriz_a, matriz_b, label='data', color='blue')
                plt.title('Regresión líneal')
                regresion_lineal = linear_model.LinearRegression()
                regresion_lineal.fit(matriz_a.reshape(-1, 1), matriz_b)
                print('\nParámetros del modelo de regresión')
                print('b (Pendiente) = ' + str(regresion_lineal.coef_) + ', a (Punto de corte) = ' + str(
                    regresion_lineal.intercept_))

                # vamos a predecir el periodo 7 (z = [7]
                pronostico = regresion_lineal.predict(z.reshape(-1, 1))

                print('\nPronósticos')
                for i in range(len(z)):
                    print('Pronóstico para el periodo {0} = {1} '.format(z[i], pronostico[i]))

                pronostico_entrenamiento = regresion_lineal.predict(matriz_a.reshape(-1, 1))
                mse = mean_squared_error(y_true=matriz_b, y_pred=pronostico_entrenamiento)
                rmse = np.sqrt(mse)
                print('\nEvaluación de calidad de la regresión')
                print('Error Cuadrático Medio (MSE) = ' + str(mse))
                print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))

                r2 = regresion_lineal.score(matriz_a.reshape(-1, 1), matriz_b)
                print('Coeficiente de Determinación R2 = ' + str(r2))

                plt.plot(matriz_a, pronostico_entrenamiento, label='data', color='red')
                plt.xlabel('Experimento A')
                plt.ylabel('Experimento B')




                print()
                fig, ax = plt.subplots()
                ax.scatter(experimento_a, experimento_b)
                plt.title("Diagrama de dispersión")
                plt.show()
                print()






                print()
            elif opcion2 == 4:
                salir2 = True
            else:
                print("Digíte una opción válida")

    elif opcion1 == 3:
        while not salir2:
            print("1. Graficar la función de densidad de una distribución uniforme. ")
            print("2. Graficar la función de densidad de una distribución Bernoulli. ")
            print("3. Graficar la función de densidad de una distribución Poisson. ")
            print("4. Graficar la funciónde densidad de una distribución Exponencial")
            print("5. Salir")
            print()
            opcion2 = int(input("Ingrese una opción: "))
            if opcion2 == 1:
                print()
                print()



                uni = np.random.uniform(low=4.0, high=2.0, size=10)


                print(uni)
                data_df = pd.DataFrame(uni)
                data_df.plot.density(color='green')
                plt.title('Gráfico de densidad para una distribución Uniforme')
                plt.show()
                print()
            elif opcion2 == 2:
                print()
                print()

                p = 0.3
                mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')

                r = bernoulli.rvs(p, size=1000)

                dta = pd.DataFrame(r)


                dta.plot.density(color='green')
                plt.title('Gráfico de densidad para una distribución Bernoulli')
                plt.show()
                print()
            elif opcion2 == 3:
                print()
                print()


                pois = np.random.poisson(lam=2.0, size=10)

                print(pois)
                data_df = pd.DataFrame(pois)
                data_df.plot.density(color='green')
                plt.title('Gráfico de densidad para una distribución Poisson')
                plt.show()

                print()
            elif opcion2 == 4:  #0.01, 0.99    3.6
                print()
                print()

                expo = np.random.exponential(scale=2.0, size=10)

                print(expo)
                data_df = pd.DataFrame(expo)
                data_df.plot.density(color='green')
                plt.title('Gráfico de densidad para una distribución exponencial')
                plt.show()

                print()
            elif opcion2 == 5:
                salir2 = True
            else:
                print("Digíte una opción válida")
    elif opcion1 == 4:
        salir1 = True
        print("Fin del programa")
    else:
        print("Digíte una opción válida")


