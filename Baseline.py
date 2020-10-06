# Librerías
import numpy as np # algebra lineal

import pandas as pd # manipulacion de datos
import matplotlib.pyplot as plt # graficas flexibles
import seaborn as sns # graficas comunes
from scipy.stats import variation # coeficiente de variación

import pdb # librería para hacer debugging si es necesario
import warnings

pd.set_option('display.max_rows', 200) # Ver 150 filas en celda
# %matplotlib inline

"""#F_E"""

DataFrame_filtrado = pd.read_csv("Data_Base\Alpha 1.csv")

DataFrame_filtrado_CMPX2 = DataFrame_filtrado.copy()

"""---
Boxplot
"""

plt.style.use('seaborn')
DataFrame_filtrado_CMPX2.boxplot(column='CMPX DE PARO POR URDIMBRE',figsize=(5 , 15) )

##Se hace el filtado del CMPX en los rangos comunes a suceder
DataFrame_filtrado_CMPX2 = DataFrame_filtrado_CMPX2.loc[ (0 < DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE'])
                                                         & (DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE']<7.5)
                                                        ]
DataFrame_filtrado_CMPX2.shape

DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE'] = DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE']

DataFrame_filtrado = DataFrame_filtrado_CMPX2.copy()

"""---"""

import pandas as pd
Performance = {'Estructura':['XGBRegressor' , 'Random_Forest' 
              , 'CatBoost' , 'LGMRegressor', 'NN' , 'NN_ENSAMBLE']}

list_predictors = ['RPM','MAT_U','MAT_T','TITULO_U','NUM_CABOS_U',
                   'MEZCLA_U','MEZCLA_T1','HILATURA_U','HILATURA_T1',
                   'ES_PEINADO_NO_CARDADO_U','ES_PEINADO_NO_CARDADO_T1','LIGAMENTO_FONDO','LIGAMENTO_ORILLO',
                   'DIENTES/CM_PEINE','HILOS/DIENTE_FONDO','HILOS/DIENTE_ORILLO','ANCHO_PEINE','ANCHO_CRUDO',
                   '%E_URDIMBRE','CUADROS_FONDO',
                   'PORC_HILOS_FONDO',
                   'TOTAL_HILOS/ANCHO_CRUDO','PASADAS/CM_T1','PORC_PASADAS/CM_T1' ,'RATIO_CONS1_CONS2',
                   'GR/MTL_U','GR/MTL_T1','TOTAL_PASADAS',
                   'PORC_GR/MTL_U','PORC_GR/MTL_T1','TOTAL_GR/MTL','FACTOR_NORMA', 'MAQUINA_PINZAS',
                   'NUM_COLORES_U',
                   'NUM_COLORES_T','AGUA    ',
                   'LUMINOSIDAD_T_1', 'LUMINOSIDAD_U_1', 'LUMINOSIDAD_T_2', 'LUMINOSIDAD_U_2', 'LUMINOSIDAD_T_3',
                   'LUMINOSIDAD_U_3', 'LUMINOSIDAD_T_4', 'LUMINOSIDAD_U_4', 'LUMINOSIDAD_T_5', 'LUMINOSIDAD_U_5',
                   'LUMINOSIDAD_T_6', 'LUMINOSIDAD_U_6', 
                   'COLORANTE_BLANCO_T','COLORANTE_BLANCO_U',
                   'COLORANTE_CRUDO_T','COLORANTE_CRUDO_U',
                   'COLORANTE_OTROS_T','COLORANTE_OTROS_U',
                   'COLORANTE_REACTIVO_T','COLORANTE_REACTIVO_U',
                   'COLORANTE_TINA_T','COLORANTE_TINA_U',
                   'FACT_COB_U', 'FACT_COB_T', 'PORC_FACT_COB_U',
                   'FACT_COB_TOTAL_REAL', 'TUPIDEZ',
                   'Ne_prom','CV% Ne_prom','cN/tex_prom','TPI_prom','FT_prom','CV% TPI_prom','E%_prom','CV% E_prom','CV%R_prom','CVm%_prom','I_prom','PD(-40%)_prom',
                   'PD(-50%)_prom','PG(+35%)_prom','PG(+50%)_prom','NEPS(+140%)_prom','NEPS(+200%)_prom','H_prom','Sh_prom',
                   'var_Ne_prom','var_cN/tex_prom','var_TPI_prom','var_E%_prom',
                   '%falla_E_prom','Ne_std','CV% Ne_std','cN/tex_std','TPI_std','FT_std','CV% TPI_std','E%_std','CV% E_std','CV%R_std','CVm%_std','I_std','PD(-40%)_std',
                   'PD(-50%)_std','PG(+35%)_std','PG(+50%)_std','NEPS(+140%)_std','NEPS(+200%)_std','H_std','Sh_std',
                   'var_Ne_std','var_cN/tex_std','var_TPI_std','var_E%_std',
                   '%falla_E_std','%falla_E_075','%falla_R_075','CV% E_075','CV% Ne_075','CV% TPI_075',
                   'CV%R_075','CVm%_075','E%_075','FT_075','H_075','I_075','NEPS(+140%)_075','NEPS(+200%)_075','Ne_075','PD(-40%)_075','PD(-50%)_075','PG(+35%)_075',
                   'PG(+50%)_075','Sh_075','TPI_075','cN/tex_075',
                   'var_E%_075','var_Ne_075','var_TPI_075','var_cN/tex_075',
                   '%falla_E_025','%falla_R_025','CV% E_025','CV% Ne_025','CV% TPI_025','CV%R_025','CVm%_025','E%_025','FT_025','H_025','I_025','NEPS(+140%)_025',
                   'NEPS(+200%)_025','Ne_025','PD(-40%)_025','PD(-50%)_025','PG(+35%)_025','PG(+50%)_025','Sh_025','TPI_025','cN/tex_025',
                   'var_E%_025','var_Ne_025','var_TPI_025','var_cN/tex_025' ]

list_targets = ['CMPX DE PARO POR URDIMBRE']

predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]

#quitar variables correlacionadas
df=DataFrame_filtrado.copy()
# Create correlation matrix
corr_matrix = df[predictores_categoricos+predictores_numericos].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
for i in to_drop:
  if i in predictores_numericos:
    predictores_numericos.remove(i)
  elif i in predictores_categoricos:
    predictores_categoricos.remove(i)
# df.drop(df[to_drop], axis=1,inplace=True)

"""Separamos las variables para entrenamiento y validacion"""

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

"""Tratamiento de variables categoricas"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Aplicamos hot enconder en los datos categoricos
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[predictores_categoricos].astype(str)))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[predictores_categoricos].astype(str)))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train[predictores_numericos]
num_X_test = X_test[predictores_numericos]

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

#Renombramos a las columnas
lista_nombres_numericos = num_X_train[predictores_numericos].columns 
lista_nombres_categoricos = OH_encoder.get_feature_names(predictores_categoricos)
lista_nombres = list(lista_nombres_numericos) + list(lista_nombres_categoricos)
OH_X_train.columns = lista_nombres
OH_X_test.columns = lista_nombres

#Borrar
OH_X_train

"""#Primer Lote de Modelos

## XGBOOSTRegressor
"""

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost
set_to_eval = [(OH_X_train, y_train), (OH_X_test, y_test)]

my_model = XGBRegressor(objective='reg:squarederror' , n_estimators=500 , learning_rate=0.01 , max_depth= 8)
my_model_grid = my_model.fit(OH_X_train, y_train,
             early_stopping_rounds=2, 
             eval_metric=["error", "mae"],
             eval_set= set_to_eval ,
             verbose=2, 
             )

results = my_model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Error en la Prediccion')
plt.xlabel('error')
plt.title('Error en el XGBRegressor')
plt.show()
# plot mae
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mae'], label='Train')
ax.plot(x_axis, results['validation_1']['mae'], label='Test')
ax.legend()
plt.ylabel('MAE')
plt.title('XGBoost Classification MAE')
plt.show()

test_resultados = my_model.predict(OH_X_test).flatten()

fig, ax = plt.subplots()
ax.scatter(y_test, test_resultados)
plt.style.use('seaborn')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'go--', lw=4)
ax.set_xlabel('Real')
ax.set_ylabel('Predictor')
plt.figure(figsize= (20,20))
plt.show()

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)
plt.xlabel('Error CMPX_U')

from sklearn.metrics import r2_score , mean_absolute_error
predictions=my_model.predict(OH_X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))

Resultado=r2_score(y_test,my_model.predict(OH_X_test))
print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado*100))

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

XGBMAPE = mean_absolute_percentage_error(y_test,predictions)

Performance['MAE']= [str(mean_absolute_error(predictions, y_test))[0:5]]
Performance['MAPE'] = [str((XGBMAPE))[0:5]]
Performance['R2'] = [str((Resultado*100))[0:5]]

import shap
xgboost.plot_importance(my_model , max_num_features=20)

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(OH_X_test.iloc[0])
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, OH_X_test.iloc[0])

#Display
pd.DataFrame(data=shap_values, columns=OH_X_test.columns, index=[0]).transpose().sort_values(by=0, ascending=True)

#USAREMOS LAS PREDICCIONES DEL RANDOM FOREST COMO OTRA COLUMNA PARA LA NN
X_cols_train = pd.DataFrame(OH_encoder.transform(X[predictores_categoricos].astype(str)))
X_cols_train.index = X.index
num_X = X[predictores_numericos]
X_total = pd.concat([num_X, X_cols_train], axis=1)
#Renombramos a las columnas
lista_nombres_numericos = num_X[predictores_numericos].columns 
lista_nombres_categoricos = OH_encoder.get_feature_names(predictores_categoricos)
lista_nombres = list(lista_nombres_numericos) + list(lista_nombres_categoricos)
X_total.columns = lista_nombres
X_total

#Predecimos
predictions_to_NN_1=my_model.predict(X_total)
len(predictions_to_NN_1)
Tabla_Explicativa = pd.Series(predictions_to_NN_1)

"""####Permutation test"""

#import eli5

#permutador = eli5.sklearn.PermutationImportance(my_model).fit(OH_X_test[list(OH_X_test.columns)], y_test)
#eli5.explain_weights(permutador, feature_names=list(OH_X_test.columns))

#DF para explicar los datos
#D_F_Permutation = eli5.explain_weights_df(permutador,top = 20 , feature_names=list(OH_X_test.columns))

"""

#Buscamos Nuevos modelos

##Random forest
"""
list_predictors = ['RPM','MAT_U','MAT_T','TITULO_U','NUM_CABOS_U',
                   'MEZCLA_U','MEZCLA_T1','HILATURA_U','HILATURA_T1',
                   'ES_PEINADO_NO_CARDADO_U','ES_PEINADO_NO_CARDADO_T1','LIGAMENTO_FONDO','LIGAMENTO_ORILLO',
                   'DIENTES/CM_PEINE','HILOS/DIENTE_FONDO','HILOS/DIENTE_ORILLO','ANCHO_PEINE','ANCHO_CRUDO',
                   '%E_URDIMBRE','CUADROS_FONDO',
                   'PORC_HILOS_FONDO',
                   'TOTAL_HILOS/ANCHO_CRUDO','PASADAS/CM_T1','PORC_PASADAS/CM_T1' ,'RATIO_CONS1_CONS2',
                   'GR/MTL_U','GR/MTL_T1','TOTAL_PASADAS',
                   'PORC_GR/MTL_U','PORC_GR/MTL_T1','TOTAL_GR/MTL','FACTOR_NORMA', 'MAQUINA_PINZAS',
                   'NUM_COLORES_U',
                   'NUM_COLORES_T','AGUA    ',
                   'LUMINOSIDAD_T_1', 'LUMINOSIDAD_U_1', 'LUMINOSIDAD_T_2', 'LUMINOSIDAD_U_2', 'LUMINOSIDAD_T_3',
                   'LUMINOSIDAD_U_3', 'LUMINOSIDAD_T_4', 'LUMINOSIDAD_U_4', 'LUMINOSIDAD_T_5', 'LUMINOSIDAD_U_5',
                   'LUMINOSIDAD_T_6', 'LUMINOSIDAD_U_6', 
                   'COLORANTE_BLANCO_T','COLORANTE_BLANCO_U',
                   'COLORANTE_CRUDO_T','COLORANTE_CRUDO_U',
                   'COLORANTE_OTROS_T','COLORANTE_OTROS_U',
                   'COLORANTE_REACTIVO_T','COLORANTE_REACTIVO_U',
                   'COLORANTE_TINA_T','COLORANTE_TINA_U',
                   'FACT_COB_U', 'FACT_COB_T', 'PORC_FACT_COB_U',
                   'FACT_COB_TOTAL_REAL', 'TUPIDEZ',
                   'Ne_prom','CV% Ne_prom','cN/tex_prom','TPI_prom','FT_prom','CV% TPI_prom','E%_prom','CV% E_prom','CV%R_prom','CVm%_prom','I_prom','PD(-40%)_prom',
                   'PD(-50%)_prom','PG(+35%)_prom','PG(+50%)_prom','NEPS(+140%)_prom','NEPS(+200%)_prom','H_prom','Sh_prom',
                   'var_Ne_prom','var_cN/tex_prom','var_TPI_prom','var_E%_prom',
                   '%falla_E_prom','Ne_std','CV% Ne_std','cN/tex_std','TPI_std','FT_std','CV% TPI_std','E%_std','CV% E_std','CV%R_std','CVm%_std','I_std','PD(-40%)_std',
                   'PD(-50%)_std','PG(+35%)_std','PG(+50%)_std','NEPS(+140%)_std','NEPS(+200%)_std','H_std','Sh_std',
                   'var_Ne_std','var_cN/tex_std','var_TPI_std','var_E%_std',
                   '%falla_E_std','%falla_E_075','%falla_R_075','CV% E_075','CV% Ne_075','CV% TPI_075',
                   'CV%R_075','CVm%_075','E%_075','FT_075','H_075','I_075','NEPS(+140%)_075','NEPS(+200%)_075','Ne_075','PD(-40%)_075','PD(-50%)_075','PG(+35%)_075',
                   'PG(+50%)_075','Sh_075','TPI_075','cN/tex_075',
                   'var_E%_075','var_Ne_075','var_TPI_075','var_cN/tex_075',
                   '%falla_E_025','%falla_R_025','CV% E_025','CV% Ne_025','CV% TPI_025','CV%R_025','CVm%_025','E%_025','FT_025','H_025','I_025','NEPS(+140%)_025',
                   'NEPS(+200%)_025','Ne_025','PD(-40%)_025','PD(-50%)_025','PG(+35%)_025','PG(+50%)_025','Sh_025','TPI_025','cN/tex_025',
                   'var_E%_025','var_Ne_025','var_TPI_025','var_cN/tex_025' ]

list_targets = ['CMPX DE PARO POR URDIMBRE']

predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]

#quitar variables correlacionadas
df=DataFrame_filtrado.copy()
# Create correlation matrix
corr_matrix = df[predictores_categoricos+predictores_numericos].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
for i in to_drop:
  if i in predictores_numericos:
    predictores_numericos.remove(i)
  elif i in predictores_categoricos:
    predictores_categoricos.remove(i)
# df.drop(df[to_drop], axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Aplicamos hot enconder en los datos categoricos
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[predictores_categoricos].astype(str)))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[predictores_categoricos].astype(str)))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train[predictores_numericos]
num_X_test = X_test[predictores_numericos]

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

#Renombramos a las columnas
lista_nombres_numericos = num_X_train[predictores_numericos].columns 
lista_nombres_categoricos = OH_encoder.get_feature_names(predictores_categoricos)
lista_nombres = list(lista_nombres_numericos) + list(lista_nombres_categoricos)
OH_X_train.columns = lista_nombres
OH_X_test.columns = lista_nombres

# Importamos el random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 500 , criterion = 'mse', random_state = 42)# Entrenamos el modelo con 1000 arboles

rf.fit(OH_X_train, y_train)

# Predecimos
predictions_to_NN_2=rf.predict(X_total)

predictions = rf.predict(OH_X_test)

errors_rf = abs(predictions - y_test)

mape_rf = mean_absolute_percentage_error(y_test, predictions)

Resultado_rf=r2_score(y_test,rf.predict(OH_X_test))
print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado_rf*100))

Performance['MAE'].append(str(round(np.mean(errors_rf))))
Performance['MAPE'].append(str((mape_rf))[0:5])
Performance['R2'].append(str((Resultado_rf*100))[0:5])

test_resultados = rf.predict(OH_X_test).flatten()
plt.figure(figsize= (20,20))
fig, ax = plt.subplots()
ax.scatter(y_test, test_resultados)
plt.style.use('seaborn')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'go--', lw=4)
ax.set_xlabel('Real')
ax.set_ylabel('Predictor')
plt.show()

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)

"""##CatBoostRegressor

Para evitar el aumento sustancial de columnas por el uso de one hot encoder en las varibales categoricas se usaran 2 diferentes modelos que no requieren este tratamiento
"""

# Revisamos la version
import catboost
from sklearn.model_selection import GridSearchCV
print(catboost.__version__)


predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]

#quitar variables correlacionadas
df=DataFrame_filtrado.copy()
# Create correlation matrix
corr_matrix = df[predictores_categoricos+predictores_numericos].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
for i in to_drop:
  if i in predictores_numericos:
    predictores_numericos.remove(i)
  elif i in predictores_categoricos:
    predictores_categoricos.remove(i)
# df.drop(df[to_drop], axis=1,inplace=True)

len(predictores_numericos + predictores_categoricos)

print(predictores_categoricos)

"""Separamos las variables para entrenamiento y validacion"""

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

"""Tratamiento de variables"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Aplicamos hot enconder en los datos categoricos

OH_cols_train = pd.DataFrame(X_train[predictores_categoricos].astype(str))
OH_cols_test = pd.DataFrame(X_test[predictores_categoricos].astype(str))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train[predictores_numericos]
num_X_test = X_test[predictores_numericos]

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


parameters = {'depth'         : [6,8,10],
              'learning_rate' : [0.01, 0.05, 0.1],
              'l2_leaf_reg': [1,4,9],
              'iterations'    : [30, 50, 100]
              }

#Creamos el modelo
#model_CBR = catboost.CatBoostRegressor(cat_features= OH_X_train[predictores_categoricos] ,eval_metric= 'MAPE')
#Busqueda en grilla
#grid = GridSearchCV(estimator=model_CBR, param_grid = parameters, cv = 2, verbose= 1)
#grid.fit(OH_X_train, y_train)
#Mostrar los resultados
#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
#print("\n The best score across ALL searched params:\n", grid.best_score_)
#print("\n The best parameters across ALL searched params:\n", grid.best_params_)

"""The best parameters across ALL searched params:
 {'depth': 10, 'iterations': 100, 'l2_leaf_reg': 1, 'learning_rate': 0.05}
"""

CBR = catboost.CatBoostRegressor(cat_features= OH_X_train[predictores_categoricos] ,
                                 learning_rate= 0.05 , depth= 10 , iterations= 100 ,eval_metric= 'MAPE' 
                                 ,l2_leaf_reg = 1)

CBR.fit(OH_X_train, y_train)

###
#Columnas para la red neuronal
###
X_cols = pd.DataFrame(X[predictores_categoricos].astype(str))
X_cols.index = X.index
num_X = X[predictores_numericos]
X_temp = pd.concat([num_X, X_cols], axis=1)

predictions_to_NN_3=CBR.predict(X_temp)
#

predictions = CBR.predict(OH_X_test)
errors = abs(predictions - y_test)
Resultado=r2_score(y_test,CBR.predict(OH_X_test))

print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado*100))

print("El MAPE de esta solucion fue de: {:6.2f} %".format(mean_absolute_percentage_error(y_test,predictions)))

test_resultados = CBR.predict(OH_X_test).flatten()
fig, ax = plt.subplots()
ax.scatter(y_test, test_resultados)
plt.style.use('seaborn')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'go--', lw=4)
ax.set_xlabel('Real')
ax.set_ylabel('Predictor')
plt.figure(figsize= (20,20))
plt.show()

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)
plt.xlabel('Error CMPX_U')

Performance['MAE'].append(str(round(np.mean(errors))))
Performance['MAPE'].append(str((mean_absolute_percentage_error(y_test,predictions)))[0:5])
Performance['R2'].append(str((Resultado*100))[0:5])

"""##LGBMRegressor"""

# check lightgbm version
import lightgbm
print(lightgbm.__version__)

P_total= predictores_numericos + predictores_categoricos

"""Separamos las variables para entrenamiento y validacion"""

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

"""Tratamiento de variables"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Aplicamos hot enconder en los datos categoricos

OH_cols_train = pd.DataFrame(X_train[predictores_categoricos].astype(str))
OH_cols_test = pd.DataFrame(X_test[predictores_categoricos].astype(str))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train[predictores_numericos]
num_X_test = X_test[predictores_numericos]

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

params = {
    'num_leaves': [7, 14, 21, 28],
    'learning_rate': [0.1, 0.03, 0.003],
    'max_depth': [-1, 3, 5],
    'n_estimators': [50, 100, 200],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}

#Creamos el modelo
#model_LGB = lightgbm.LGBMRegressor(eval_metric= 'MAPE'  )
#Busqueda en grilla
#grid = GridSearchCV(estimator=model_LGB, param_grid = params, cv = 2, verbose= 1 , n_jobs= -1)
#grid.fit(num_X_train, y_train)
#Mostrar los resultados
#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
#print("\n The best score across ALL searched params:\n", grid.best_score_)
#print("\n The best parameters across ALL searched params:\n", grid.best_params_)

"""The best parameters across ALL searched params:
 {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': -1, 'min_split_gain': 0.3, 'n_estimators': 200, 'num_leaves': 21, 'reg_alpha': 1.3, 'reg_lambda': 1.1, 'subsample': 0.9, 'subsample_freq': 20}
"""

LGB = lightgbm.LGBMRegressor(colsample_bytree= 0.7 , num_leaves= 21 , reg_alpha= 1.3 , n_estimators=200 
                             ,learning_rate = 0.03 , max_depth = -1 ,min_split_gain =0.3 ,reg_lambda = 1.1
                              ,subsample = 0.9, subsample_freq = 20 )

for feature in predictores_categoricos:
    OH_X_train[feature] = pd.Series(OH_X_train[feature], dtype="category")
    OH_X_test[feature] = pd.Series(OH_X_train[feature], dtype="category")

LGB.fit(OH_X_train, y_train  )

predictions = LGB.predict(OH_X_test)
errors = abs(predictions - y_test)
Resultado=r2_score(y_test,LGB.predict(OH_X_test))

#
#Columnas para la red neuronal
#
X_temp = X.copy()
for feature in predictores_categoricos:
    X_temp[feature] = pd.Series(X_temp[feature], dtype="category")
    X_temp[feature] = pd.Series(X_temp[feature], dtype="category")

predictions_to_NN_4=LGB.predict(X_temp[predictores_categoricos + predictores_numericos])
#

print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado*100))

print("El MAPE de esta solucion fue de: {:6.2f} %".format(mean_absolute_percentage_error(y_test,predictions)))

test_resultados = LGB.predict(OH_X_test).flatten()

fig, ax = plt.subplots()
ax.scatter(y_test, test_resultados)
plt.style.use('seaborn')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'go--', lw=4)
ax.set_xlabel('Real')
ax.set_ylabel('Predictor')
plt.figure(figsize= (20,20))
plt.show()

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)
plt.xlabel('Error CMPX_U')

Performance['MAE'].append(str(round(np.mean(errors))))
Performance['MAPE'].append(str((mean_absolute_percentage_error(y_test,predictions)))[0:5])
Performance['R2'].append(str((Resultado*100))[0:5])


"""## Redes neuronales."""

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Aplicamos hot enconder en los datos categoricos
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[predictores_categoricos].astype(str)))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[predictores_categoricos].astype(str)))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train.drop(predictores_categoricos, axis=1)
num_X_test = X_test.drop(predictores_categoricos, axis=1)

# se escalan los datos numéricos
scaler = StandardScaler()
Num_X_Scaler_train=pd.DataFrame(scaler.fit_transform(num_X_train[predictores_numericos]))
Num_X_Scaler_test=pd.DataFrame(scaler.fit_transform(num_X_test[predictores_numericos]))
#elimina los indices asi que los volvemos a poner
Num_X_Scaler_train.index = X_train.index
Num_X_Scaler_test.index = X_test.index


# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([Num_X_Scaler_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([Num_X_Scaler_test, OH_cols_test], axis=1)

lista_nombres_numericos = num_X_train[predictores_numericos].columns 
lista_nombres_categoricos = OH_encoder.get_feature_names(predictores_categoricos)
lista_nombres = list(lista_nombres_numericos) + list(lista_nombres_categoricos)
OH_X_train.columns = lista_nombres
OH_X_test.columns = lista_nombres

#Imprimimos
print(OH_X_train)

#PCA
from sklearn.decomposition import PCA
# 80% de varianza explicada
#pca = PCA(.80)
#pca.fit(OH_X_train)
#Se aplica PCA a la data a analizar
#X_red_train = pca.transform(OH_X_train)
#X_red_test = pca.transform(OH_X_test)

#X_red_train = pd.DataFrame(X_red_train)
#X_red_test = pd.DataFrame(X_red_test)

#Se le asigna el indice
#X_red_train.index = X_train.index
#X_red_test.index = X_test.index
######################################################
#En caso de que se quiera evitar usar PCA
######################################################
X_red_train = OH_X_train
X_red_test = OH_X_test
print( X_red_test)
#print(pca.explained_variance_)

#---------------------------------------------------- red (Tensor)------------------------------
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping


K.regularizers.l1(0.01)
K.regularizers.l2(0.01)
K.regularizers.l1_l2(l1=0.01, l2=0.01)


model = K.Sequential([
    K.layers.Dense(32, activation='relu',kernel_initializer='he_normal', input_shape=(X_red_train.shape[1],)),
    K.layers.Dropout(0.2),
    K.layers.Dense(16, activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.03) ),
    K.layers.Dropout(0.2),
    K.layers.Dense(8, activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.03)),
    K.layers.Dropout(0.05),
    K.layers.Dense(1 , activation = 'linear')
    ])
model.summary() # resumen de la arquitectura
K.backend.set_epsilon(1)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='MSE', 
              metrics = ['MAE', 'MAPE' ,'MSE'])

K.backend.set_epsilon(1)
history=model.fit(X_red_train,y_train,validation_split=0.35,batch_size=100 ,epochs=200,verbose=1 , callbacks= EarlyStopping(monitor = 'val_MAPE' , patience = 10 , mode = "min"))

K.backend.set_epsilon(1)
score=model.evaluate(X_red_test,y_test,batch_size=100)

# summarize history for loss
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


plt.plot(history.history['loss'] , color = "blue")
plt.plot(history.history['val_loss'], color= "red")
plt.grid()
plt.title('Como se desempeño el modelo a travez de las épocas')
plt.ylabel('Error en la predicción')
plt.xlabel('Épocas')
plt.legend(['Predicciones en la etapa entrenamiento', 'Predicciones en la estapa de evaluacion'], loc='upper left')
plt.show()

Resultado=r2_score(y_test,model.predict(X_red_test,batch_size=100))
print("¿Que tan acertada es la solución?: {:6.2f} %".format(Resultado*100))

# Siempre es útil visualizar el historial de entrenamiento para asegurarnos que el modelo no sobreajuste!
historial = pd.DataFrame(history.history)

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
historial[['MSE', 'val_MSE']].plot(ax=ax1)
historial[['MAE', 'val_MAE']].plot(ax=ax2)
historial[['MAPE', 'val_MAPE']].plot(ax=ax3)

ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax3.set_xlabel('Epochs')

test_resultados = model.predict(X_red_test).flatten()

fig, ax = plt.subplots()
ax.scatter(y_test, test_resultados)
plt.style.use('seaborn')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'go--', lw=4)
ax.set_xlabel('Real')
ax.set_ylabel('Predictor')
plt.figure(figsize= (20,20))
plt.show()

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)
plt.xlabel('Error CMPX_U')

NN_A_MAPE  = historial.loc[ historial.shape[0] -1 ,'val_MAPE' ]

from sklearn.metrics import r2_score
Resultado=r2_score(y_test,model.predict(X_red_test,batch_size=100))
print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado*100))

Performance['MAE'].append(str(historial.loc[ historial.shape[0] -1 ,'MAE' ])[0:5])
Performance['MAPE'].append(str((NN_A_MAPE))[0:5])
Performance['R2'].append(str((Resultado*100))[0:5])



"""## Redes neuronales con prediccion ensamblada

*Se le agregan las predicciones del XGBRegressor Y se entrenan a las NN*
"""
list_predictors = ['RPM','MAT_U','MAT_T','TITULO_U','NUM_CABOS_U',
                   'MEZCLA_U','MEZCLA_T1','HILATURA_U','HILATURA_T1',
                   'ES_PEINADO_NO_CARDADO_U','ES_PEINADO_NO_CARDADO_T1','LIGAMENTO_FONDO','LIGAMENTO_ORILLO',
                   'DIENTES/CM_PEINE','HILOS/DIENTE_FONDO','HILOS/DIENTE_ORILLO','ANCHO_PEINE','ANCHO_CRUDO',
                   '%E_URDIMBRE','CUADROS_FONDO',
                   'PORC_HILOS_FONDO',
                   'TOTAL_HILOS/ANCHO_CRUDO','PASADAS/CM_T1','PORC_PASADAS/CM_T1' ,'RATIO_CONS1_CONS2',
                   'GR/MTL_U','GR/MTL_T1','TOTAL_PASADAS',
                   'PORC_GR/MTL_U','PORC_GR/MTL_T1','TOTAL_GR/MTL','FACTOR_NORMA', 'MAQUINA_PINZAS',
                   'NUM_COLORES_U',
                   'NUM_COLORES_T','AGUA    ',
                   'LUMINOSIDAD_T_1', 'LUMINOSIDAD_U_1', 'LUMINOSIDAD_T_2', 'LUMINOSIDAD_U_2', 'LUMINOSIDAD_T_3',
                   'LUMINOSIDAD_U_3', 'LUMINOSIDAD_T_4', 'LUMINOSIDAD_U_4', 'LUMINOSIDAD_T_5', 'LUMINOSIDAD_U_5',
                   'LUMINOSIDAD_T_6', 'LUMINOSIDAD_U_6', 
                   'COLORANTE_BLANCO_T','COLORANTE_BLANCO_U',
                   'COLORANTE_CRUDO_T','COLORANTE_CRUDO_U',
                   'COLORANTE_OTROS_T','COLORANTE_OTROS_U',
                   'COLORANTE_REACTIVO_T','COLORANTE_REACTIVO_U',
                   'COLORANTE_TINA_T','COLORANTE_TINA_U',
                   'FACT_COB_U', 'FACT_COB_T', 'PORC_FACT_COB_U',
                   'FACT_COB_TOTAL_REAL', 'TUPIDEZ',
                   'Ne_prom','CV% Ne_prom','cN/tex_prom','TPI_prom','FT_prom','CV% TPI_prom','E%_prom','CV% E_prom','CV%R_prom','CVm%_prom','I_prom','PD(-40%)_prom',
                   'PD(-50%)_prom','PG(+35%)_prom','PG(+50%)_prom','NEPS(+140%)_prom','NEPS(+200%)_prom','H_prom','Sh_prom',
                   'var_Ne_prom','var_cN/tex_prom','var_TPI_prom','var_E%_prom',
                   '%falla_E_prom','Ne_std','CV% Ne_std','cN/tex_std','TPI_std','FT_std','CV% TPI_std','E%_std','CV% E_std','CV%R_std','CVm%_std','I_std','PD(-40%)_std',
                   'PD(-50%)_std','PG(+35%)_std','PG(+50%)_std','NEPS(+140%)_std','NEPS(+200%)_std','H_std','Sh_std',
                   'var_Ne_std','var_cN/tex_std','var_TPI_std','var_E%_std',
                   '%falla_E_std','%falla_E_075','%falla_R_075','CV% E_075','CV% Ne_075','CV% TPI_075',
                   'CV%R_075','CVm%_075','E%_075','FT_075','H_075','I_075','NEPS(+140%)_075','NEPS(+200%)_075','Ne_075','PD(-40%)_075','PD(-50%)_075','PG(+35%)_075',
                   'PG(+50%)_075','Sh_075','TPI_075','cN/tex_075',
                   'var_E%_075','var_Ne_075','var_TPI_075','var_cN/tex_075',
                   '%falla_E_025','%falla_R_025','CV% E_025','CV% Ne_025','CV% TPI_025','CV%R_025','CVm%_025','E%_025','FT_025','H_025','I_025','NEPS(+140%)_025',
                   'NEPS(+200%)_025','Ne_025','PD(-40%)_025','PD(-50%)_025','PG(+35%)_025','PG(+50%)_025','Sh_025','TPI_025','cN/tex_025',
                   'var_E%_025','var_Ne_025','var_TPI_025','var_cN/tex_025' ]

list_targets = ['CMPX DE PARO POR URDIMBRE']

predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]

#quitar variables correlacionadas
df=DataFrame_filtrado.copy()
# Create correlation matrix
corr_matrix = df[predictores_categoricos+predictores_numericos].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
for i in to_drop:
  if i in predictores_numericos:
    predictores_numericos.remove(i)
  elif i in predictores_categoricos:
    predictores_categoricos.remove(i)
# df.drop(df[to_drop], axis=1,inplace=True)

DataFrame_filtrado["Predicciones_XGB"] =  predictions_to_NN_1
DataFrame_filtrado["Predicciones_RF"] =  predictions_to_NN_2
DataFrame_filtrado["Predicciones_CB"] =  predictions_to_NN_3
DataFrame_filtrado["Predicciones_LGM"] =  predictions_to_NN_4


from sklearn.model_selection import train_test_split
list_predictors = list_predictors + ["Predicciones_XGB"] + ["Predicciones_CB"] + ["Predicciones_RF"]
X=DataFrame_filtrado[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

predictores_numericos = predictores_numericos +  ["Predicciones_XGB"] + ["Predicciones_CB"] + ["Predicciones_RF"]
# Aplicamos hot enconder en los datos categoricos
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[predictores_categoricos].astype(str)))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[predictores_categoricos].astype(str)))

# One-hot encoding elimina los indices asi que los volvemos a poner
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Eliminamos las columnas categoricas de nuestra data para luego remplazarlas con las resultantes del HOE
num_X_train = X_train.drop(predictores_categoricos, axis=1)
num_X_test = X_test.drop(predictores_categoricos, axis=1)

# se escalan los datos numéricos
scaler = StandardScaler()
Num_X_Scaler_train=pd.DataFrame(scaler.fit_transform(num_X_train[predictores_numericos]))
Num_X_Scaler_test=pd.DataFrame(scaler.fit_transform(num_X_test[predictores_numericos]))
#elimina los indices asi que los volvemos a poner
Num_X_Scaler_train.index = X_train.index
Num_X_Scaler_test.index = X_test.index


# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([Num_X_Scaler_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([Num_X_Scaler_test, OH_cols_test], axis=1)

lista_nombres_numericos = num_X_train[predictores_numericos].columns 
lista_nombres_categoricos = OH_encoder.get_feature_names(predictores_categoricos)
lista_nombres = list(lista_nombres_numericos) + list(lista_nombres_categoricos)
OH_X_train.columns = lista_nombres
OH_X_test.columns = lista_nombres

#PCA
from sklearn.decomposition import PCA
# 80% de varianza explicada
#pca = PCA(.80)
#pca.fit(OH_X_train)
#Se aplica PCA a la data a analizar
#X_red_train = pca.transform(OH_X_train)
#X_red_test = pca.transform(OH_X_test)
#
#X_red_train = pd.DataFrame(X_red_train)
#X_red_test = pd.DataFrame(X_red_test)

#Se le agrega la prediccion del RF 
#X_red_train.index = X_train.index
#X_red_test.index = X_test.index

#######################################################
#En caso de que no se desee usar PCA
#######################################################
X_red_train = OH_X_train
X_red_test = OH_X_test
print( X_red_test)
#print(pca.explained_variance_)

#---------------------------------------------------- red (Tensor)------------------------------
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping


K.regularizers.l1(0.01)
K.regularizers.l2(0.01)
K.regularizers.l1_l2(l1=0.01, l2=0.01)


model = K.Sequential([
    K.layers.Dense(32, activation='relu',kernel_initializer='he_normal', input_shape=(X_red_train.shape[1],)),
    K.layers.Dropout(0.2),
    K.layers.Dense(16, activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.03) ),
    K.layers.Dropout(0.2),
    K.layers.Dense(8, activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.03)),
    K.layers.Dropout(0.05),
    K.layers.Dense(1, activation='linear')
    ])

model.summary() # resumen de la arquitectura
K.backend.set_epsilon(1)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='MAPE', 
              metrics = ['MAE', 'MAPE'])

K.backend.set_epsilon(1)
history=model.fit(X_red_train,y_train,validation_split=0.35,batch_size=100 ,epochs=500,verbose=1 , callbacks= EarlyStopping(monitor = 'val_MAPE' , patience = 5 , mode = "min"))

score=model.evaluate(X_red_test,y_test,batch_size=100)

# summarize history for loss
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


plt.plot(history.history['loss'] , color = "blue")
plt.plot(history.history['val_loss'], color= "red")
plt.grid()
plt.title('Como se desempeño el modelo a travez de las épocas')
plt.ylabel('Error en la predicción')
plt.xlabel('Épocas')
plt.legend(['Predicciones en la etapa entrenamiento', 'Predicciones en la estapa de evaluacion'], loc='upper left')
plt.show()

Resultado=r2_score(y_test,model.predict(X_red_test,batch_size=100))
print("¿Que tan acertada es la solución?: {:6.2f} %".format(Resultado*100))

# Siempre es útil visualizar el historial de entrenamiento para asegurarnos que el modelo no sobreajuste!
historial = pd.DataFrame(history.history)

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
historial[['loss', 'val_loss']].plot(ax=ax1)
historial[['MAE', 'val_MAE']].plot(ax=ax2)
historial[['MAPE', 'val_MAPE']].plot(ax=ax3)

ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax3.set_xlabel('Epochs')

from sklearn.metrics import r2_score
Resultado=r2_score(y_test,model.predict(X_red_test,batch_size=100))
print("El R2 de esta solucion fue de: {:6.2f} %".format(Resultado*100))

#Se genera un histograma con los errores de las predicciones
Error_histograma = y_test - test_resultados 
plt.figure(figsize = (15 , 5))
sns.distplot(Error_histograma , kde= False)
plt.xlabel('Error CMPX_U')
plt.show()
#background

#background = X_red_test.sample( n = 1000 , random_state= 4)
#Expositor = shap.DeepExplainer(model,data = background)

NN_CASCADA_MAPE = historial.loc[ historial.shape[0] -1 ,'val_MAPE' ]

Performance['MAE'].append(str(historial.loc[ historial.shape[0] -1 ,'MAE' ])[0:5])
Performance['MAPE'].append(str((NN_CASCADA_MAPE))[0:5])
Performance['R2'].append(str((Resultado*100))[0:5])

Performance


