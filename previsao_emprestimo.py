
import pandas as pd

#Le a base de dados
base = pd.read_csv('loan_prediction.csv')

#Exclui as linhas com valores faltantes
base.dropna(how='any',inplace=True)

previsores = base.iloc[:, 1:12].values
classe = base.iloc[:, 12].values

#Substutui dados nominais para dados discretos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 4] = labelencoder_previsores.fit_transform(previsores[:, 4])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,10])
#previsores = onehotencoder.fit_transform(previsores).toarray()

#Escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Divide a base de dados
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.20, random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix , accuracy_score
#Score de 0.7604
#Foi o melhor score, depois de varias configurações diferentes
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
