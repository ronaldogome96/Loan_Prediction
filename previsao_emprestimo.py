
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pyod.models.knn import KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def leitura():
    base = pd.read_csv('loan_prediction.csv')
    return base

def exclusao(base):
    base.dropna(how='any',inplace=True)
    base.drop('Loan_ID',axis = 1,  inplace = True)
    return base

def transform(base):
    for column in base.select_dtypes(['object']).columns:
        base[column] = pd.Categorical(base[column], categories=base[column].unique()).codes
    return base
 
def outliers(base):
    detector = KNN()
    detector.fit(base)
    previsoes = detector.labels_
    outliers = []
    for i in range(len(previsoes)):
        if previsoes[i] == 1:
            outliers.append(i)
    base = base.drop(base.index[outliers])
    return base

def parametrize(base):
    atributos = base.iloc[:, 1:11].values
    classe = base.iloc[:, 11].values
    return atributos, classe

def normalize(atributos):
    scaler = StandardScaler()
    atributos = scaler.fit_transform(atributos)
    return atributos

def treinamento(classificador):
    classificador.fit(atributos_treinamento, classe_treinamento)
    previsoes = classificador.predict(atributos_teste)
    precisao = accuracy_score(classe_teste, previsoes)
    return precisao

def redeNeural():
    classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.0000010,
                              solver = 'adam',
                              hidden_layer_sizes=(5),
                              activation='tanh')
    #Score de 0.8240
    return treinamento(classificador)

def svm():
    classificador = SVC(kernel = 'rbf', random_state=0, C=1.0)
    #Score de 0.8240
    return treinamento(classificador)

def randomForest():
    classificador = RandomForestClassifier(n_estimators = 35, criterion='entropy', random_state=0)
    #Score de 0.7870
    return treinamento(classificador)

def Knn():
    classificador = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)
    #Score de 0.8148
    return treinamento(classificador)

def naiveBayes():
    classificador = GaussianNB()
    #Score de 0.7962
    return treinamento(classificador)

def regressaoLogistica():
    classificador = LogisticRegression()
    #Score de 0.8240
    return treinamento(classificador)


base = leitura()
base = exclusao(base)
base = transform(base)
base = outliers(base)
atributos, classe = parametrize(base)
atributos = normalize(atributos)

atributos_treinamento, atributos_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.25, random_state=0)

score_RedeNeural = redeNeural()
score_SVM = svm()
score_Random_Forest = randomForest()
score_KNN = Knn()
score_Naive_Bayes = naiveBayes()
score_Regressao_Logistica = regressaoLogistica()
