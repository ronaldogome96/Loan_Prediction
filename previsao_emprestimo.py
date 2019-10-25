
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from pyod.models.knn import KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def RedeNeural():
    #Rede neural
    classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.0000010,
                              solver = 'adam',
                              hidden_layer_sizes=(5),
                              activation='tanh')
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.8240
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def SVM():
    classificador = SVC(kernel = 'rbf', random_state=0, C=1.0)
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.8240
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def RandomForest():
    classificador = RandomForestClassifier(n_estimators = 35, criterion='entropy', random_state=0)
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.7870
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def Knn():
    classificador = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.8148
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def NaiveBayes():
    classificador = GaussianNB()
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.7962
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def RegressaoLogistica():
    classificador = LogisticRegression()
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    #Score de 0.8240
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao


#Le a base de dados
base = pd.read_csv('loan_prediction.csv')

#Exclui as linhas com valores faltantes
base.dropna(how='any',inplace=True)

#Faz a detecção de outlier por esse modelo
detector = KNN()
detector.fit(base.iloc[:,6:10])

#Mostra se o dado é outlier ou nao
previsoes = detector.labels_

#Faz uma lista com as linhas que contem outliers
outliers = []
for i in range(len(previsoes)):
    #print(previsoes[i])
    if previsoes[i] == 1:
        outliers.append(i)

#Retira os outliers do data frame
base = base.drop(base.index[outliers])

previsores = base.iloc[:, 1:12].values
classe = base.iloc[:, 12].values

#Substutui dados nominais para dados discretos
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 4] = labelencoder_previsores.fit_transform(previsores[:, 4])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


#Escalonamento
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Divide a base de dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

score_RedeNeural = RedeNeural()
score_SVM = SVM()
score_Random_Forest = RandomForest()
score_KNN = Knn()
score_Naive_Bayes = NaiveBayes()
score_Regressao_Logistica = RegressaoLogistica()
