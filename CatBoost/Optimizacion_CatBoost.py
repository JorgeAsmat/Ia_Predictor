# Librerías
import numpy as np # algebra lineal

import pandas as pd # manipulacion de datos
import matplotlib.pyplot as plt # graficas flexibles
import seaborn as sns # graficas comunes
from scipy.stats import variation # coeficiente de variación

import pdb # librería para hacer debugging si es necesario
import warnings

#Fe
DataFrame_filtrado = pd.read_csv("Data_Base\Alpha 1.csv")
DataFrame_filtrado_CMPX2 = DataFrame_filtrado.copy()
DataFrame_filtrado_CMPX2 = DataFrame_filtrado_CMPX2.loc[ (0 < DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE'])
                                                         & (DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE']<7.5)
                                                        ]
DataFrame_filtrado_CMPX2.shape
DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE'] = DataFrame_filtrado_CMPX2['CMPX DE PARO POR URDIMBRE']
DataFrame_filtrado = DataFrame_filtrado_CMPX2.copy()

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
                   'var_E%_025','var_Ne_025','var_TPI_025','var_cN/tex_025' 
                  ]

list_targets = ['CMPX DE PARO POR URDIMBRE']

predictores_numericos = [i for i in list_predictors if  'float' in str(DataFrame_filtrado[i].dtype) or 'int' in str(DataFrame_filtrado[i].dtype)]
predictores_categoricos = [i for i in list_predictors if i not in predictores_numericos]

##########################
#Revisamos la correlacion
##########################
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

######################################################
#Separamos las variables para entremiento y validacion
######################################################

from sklearn.model_selection import train_test_split
X=df[list_predictors]
y=df[list_targets[0]]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.33, random_state=0)

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

#####################################
#Se comienza la optimizacion del CatBoost
#####################################

SEED = 314159265

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import r2_score
import catboost

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

def mape_objective_function(dtrain ,preds):
    labels = dtrain.get_label()
    grad = (preds - labels) / (0.2 + labels * np.abs(preds - labels))
    hess = 0.1 + np.zeros(len(preds))
    return grad, hess

def score(params):
    print("Entrenamiento con parametros: ")
    print(params)
    CatBoost_model = catboost.CatBoostRegressor(
        cat_features= OH_X_train[predictores_categoricos],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        colsample_bylevel=params['colsample_bylevel'],
        bagging_temperature=params['bagging_temperature'],
        random_strength=params['random_strength'],
        eval_metric= 'MAPE',
        l2_leaf_reg= params['l2_leaf_reg'],
    )
    CatBoost_model.fit(OH_X_train,y_train)
    predictions = CatBoost_model.predict(OH_X_test)

    score = mean_absolute_percentage_error(y_test, predictions)

    print("\tScore {0}\n\n".format(score))

    loss = score
    return {'loss': loss, 'status': STATUS_OK}

def optimize(
             #trials, 
             random_state=SEED):
    """
    Esta es  una funcion de optimizacion dado un espacio de busqueda 
    para encontrar los mejores hyperparametros de un lightgbm con un 
    evaluacion de mape
    """
    # Para evuluar los parametros de CatBoost
    # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1.0),
        'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
        'random_strength': hp.uniform('random_strength', 0.0, 100),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    }
    #Uso de fmin para encontrar los mejores hyperparametros
    best = fmin(score, space, algo=tpe.suggest, 
                # trials=trials, 
                max_evals=250)
    return best

best_hyperparams = optimize(
                            #trials
                            )
print("Los mejores hiperparametros son: ", "\n")
print(best_hyperparams) 


##################################
#Escribimos los mejores resultados
##################################
dict = best_hyperparams
f = open("Mejores_resultados_CatBoost.txt","w")
f.write( str(dict) )
f.close()    