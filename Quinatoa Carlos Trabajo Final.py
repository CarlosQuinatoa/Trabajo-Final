#Trabajo Final
#Carlos Quinatoa
#Variable: categoria_seguridad_alimentaria
#Filtro: region == "Sierra" 


#Librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statsmodels.discrete.discrete_model import Logit

#Datos
df = pd.read_csv("sample_endi_model_10p.txt", sep=";")
#eliminar datos faltantes 
df = df[~df["dcronica"].isna()]

#Filtrar dataframe con los datos de interes en este seguridad alimentaria
df.groupby('categoria_seguridad_alimentaria').size()

#1
#Reemplazamos valores 1 2 y 3 por las regiones(Nota: me pase 1 día intentandolo y reuslta que era 1.0 en lugar de 1)
df["region"] = df["region"].apply(lambda x: "Costa" if x == 1.0 else "Sierra" if x == 2.0 else "Oriente")
variables_categoricas = ['region', 'sexo', 'condicion_empleo', 'categoria_seguridad_alimentaria']
variables_numericas = ['n_hijos']
#Filtramos región sierra y Seguridad alimentaria
p_objetivo = df[(df["region"] == "Sierra") & (df['categoria_seguridad_alimentaria'] )]
#Tamaño de la población objetivo
q_infantes = len(p_objetivo)
conteo = p_objetivo['categoria_seguridad_alimentaria'].value_counts()
#Mostramos los resultados
print("Cantidad de niños en la población de seguridad alimenticia en la sierra:", q_infantes)
print("Conteo de la niños respecto a la seguridad alimenticia:",conteo)
#
#
#2
#Eliminar valores faltantes para obtener un buen reusltado en al regresión
columnas_con_nulos = ['dcronica', 'region', 'n_hijos', 'tipo_de_piso', 'espacio_lavado', 'categoria_seguridad_alimentaria', 'quintil', 'categoria_cocina', 'categoria_agua', 'serv_hig']
datos_limpios = df.dropna(subset=columnas_con_nulos)

#verificar si existen datos faltantes
print("Número de valores no finitos después de la eliminación:", datos_limpios.isna().sum())

#Convertir varibale a binaria s_a hace referencia seguridad alimentaria
datos_limpios['s_a_binario'] = datos_limpios['categoria_seguridad_alimentaria'].apply(lambda x: 1 if x == 'Seguridad' else 0)
s_a_Sierra = datos_limpios[(datos_limpios['region'] == 'Sierra') & (datos_limpios['s.a_binario'] == 1)]

#Variables de interes
variables = ['n_hijos', 'region', 'sexo', 'condicion_empleo', 'categoria_seguridad_alimentaria']

# Filtrar los datos para quedarnos solo con seguridad alimentaria "Seguridad"
for i in variables:
    s_a_Sierra  = s_a_Sierra [~s_a_Sierra [i].isna()]

# Definir las variables categóricas y numéricas
variables_categoricas = ['region', 'sexo', 'condicion_empleo']
variables_numericas = ['n_hijos']

# Crear un transformador para estandarizar las variables numéricas
transformador = StandardScaler()

# Crear una copia de los datos originales
datos_escalados = datos_limpios.copy()

# Estandarizar las variables numéricas
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Generar variables dummy en base a las categoricas
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = datos_dummies['s_a_binario']
# Definir los pesos asociados a cada observación
weights = datos_dummies['fexp_nino']

# Entrenar datos
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Asegurar que todas las variables sean numéricas
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertir a  entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Ajustar el modelo de regresión logística
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())
#
#
#
#
#
#3
# almacenar los coeficientes en un dataframe
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Tabla pivote 
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertir las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparar las predicciones con los valores reales
comparacion = (predictions_class == y_test)
# Definir el número de folds para la validación cruzada
kf = KFold(n_splits=100)
# Lista para almacenar los puntajes de precisión de cada fold
accuracy_scores = []  
# DataFrame para almacenar los coeficientes estimados en cada fold
df_params = pd.DataFrame() 


# Iterar sobre cada fold
for train_index, test_index in kf.split(X_train):
    # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustar un modelo de regresión logística en el conjunto de entrenamiento de este fold
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraer los coeficientes y organizarlos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizar predicciones en el conjunto de prueba de este fold
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calcular la precisión del modelo en el conjunto de prueba de este fold
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenar los coeficientes estimados en este fold en el DataFrame principal
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

#Precisión promedio de la validación cruzada
mean_accuracy = np.mean(accuracy_scores)
print(f"Precisión promedio de validación cruzada: {mean_accuracy}")
#Precisión promedio
precision_promedio = np.mean(accuracy_scores)

# Crear el histograma(Se crea el gráfrico)
plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio - 0.1, plt.ylim()[1] - 0.1, f'Precisión promedio: {precision_promedio:.2f}',bbox=dict(facecolor='white', alpha=0.5))
# Configurar el título y etiquetas de los ejes
plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')
plt.tight_layout()
# Mostrar el histograma
plt.show()


# Crear el histograma de los coeficientes para la variable "n_hijos"
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Calcular la media de los coeficientes para la variable "n_hijos"
media_coeficientes_n_hijos = np.mean(df_params["n_hijos"])

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(media_coeficientes_n_hijos, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(media_coeficientes_n_hijos - 0.1, plt.ylim()[1] - 0.1, f'Media de los coeficientes: {media_coeficientes_n_hijos:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar título y etiquetas de los ejes
plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()





