import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle

dados = pd.read_csv("ml_model/tabela_aluguel_completa.csv")
df = dados.copy()
df.drop(columns="Unnamed: 0", inplace=True)
df['Bairro'] = df['Bairro'].str.lstrip()
#change data type from object to category
df['Tipo']= df['Tipo'].astype('category')
df['Bairro']= df['Bairro'].astype('category')
df['Mobiliado']= df['Mobiliado'].astype('category')
#calculate the final value of the rent (20% discount) and drop old value 
df['Valor_final'] = df['Valor']*0.8
df.drop('Valor', axis=1, inplace=True)
#standardize the tipes of apartments and houses
df['Tipo'].replace(['Apartamento Flat com ...', 'Apartamento Apartamento ...', 'Apartamento Studio', 
'Apartamento Duplex', 'Apartamento Cobertura', 'Apartamento Padrão', 'Apartamento Kitchnet ...', 'Apartamento Flat sem ...',
'Apartamento Kitnet', 'Apartamento Apartamen...', 'Apartamento Flat'], ['Apartamento', 'Apartamento', 'Apartamento', 'Apartamento', 'Apartamento', 'Apartamento', 
'Apartamento', 'Apartamento', 'Apartamento', 'Apartamento', 'Apartamento'], inplace=True)
df['Tipo'].replace(['Casa Padrão', 'Casa Condomínio', 'Casa Sobrado', 'Casa Edícula'], ['Casa', 'Casa', 'Casa', 'Casa'], inplace=True)
#drop Nan values and duplicated values
df.drop(index=682, inplace=True)
df.drop(index=835, inplace=True)
df.drop_duplicates(inplace=True)
#applying log transformation
df['Valor_final'] = np.log1p(df['Valor_final'])

# Separate target from predictors
y = df['Valor_final']
X = df.drop(['Valor_final'], axis=1)


# Divide data into training and validation subsets
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Selecting categorical columns
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype.name in ["category"]] 

# Selecting numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy() #test data

# Preprocessing data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])

# Define model
# Pipeline with best hyperparameters
xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', XGBRegressor(n_estimators = 200, eval_metric= 'mae', eta = 0.1, gamma =0, random_state=42))
                                ])

# Preprocessing of training data, fit model
xgb_tuned = xgb.fit(X_train, y_train.values.ravel())

# Export model
pickle.dump(xgb_tuned, open('ml_model/model.pkl', "wb"))