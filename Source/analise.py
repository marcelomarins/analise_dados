#Análise de dados de Diabetes
import os
from numpy.core.fromnumeric import trace
from numpy.lib.utils import info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.graph_objs as go
#import plotly.offline as py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sns # vizualizar informações



plt.style.use('ggplot')

# Leitura dos diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,'data')


# List Compreension do dataset
file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]

# listando o dataset
for i in file_names:
    dataf = pd.read_csv(os.path.join(DATA_DIR, i))



print('Segue o DataSet: \n', dataf.head(10))




# *********Numero de Amostras por classe - Agua potavel = 1 e não potável =0
vl_potavel = len(dataf.loc[dataf['Potability'] == 1]) 
#classe controle
vl_naopotavel = len(dataf.loc[dataf['Potability'] == 0])

# ******************* Dados Faltantes e valores = 0 **************************************************
dt_feature = dataf.iloc[:,:-1]
dt_target = dataf.iloc[:, -1]
dt_feature = dt_feature.mask(dt_feature == 0).fillna(method='ffill')





def informacoes():
    # Apresentando informações das classes
    print(dataf['Potability'].value_counts())
    print("A quantidade de Valores Nulos",dataf.isnull().sum())



# verificando o Balanceamento dos dados das classes




def grafico_seaborn():
    # Com Seaborn
    sns.factorplot('Potability', data=dataf, kind='count')
    plt.ylabel('Variaveis')
    plt.title("Balanceamento de classes")
    plt.show()


def estatistico():
    # Plot para análise da destribuição das features e suas relações
    dataf.plot(x='ph',y='Potability',kind='scatter', title='Class and ph',color='blue')


# Análise de correlação
def plot_correlacao(corr):
    # Pega apenas a metade de baixo (simetrico)
     mask = np.zeros_like(corr, dtype=np.bool_)
     mask[np.triu_indices_from(mask, k=1)] = True
     sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.9)
     plt.show()


def correlacao2():
    corr = dataf.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.subplots(figsize=(20, 20))
    plt.show()
    


# Análise de correlação
def correlation(size=20):
    corr = dataf.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()





correlacao2()
#correlation()
#corr = dataf.corr
#plot_correlacao(corr)
#informacoes()
#grafico_seaborn()
#estatistico()