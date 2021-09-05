#imports
import os
from sys import displayhook
from numpy.core.fromnumeric import trace
from numpy.lib.utils import info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sns 
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

plt.style.use('ggplot')

# Leitura dos diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,'data')


file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]

for i in file_names:
    dataf = pd.read_csv(os.path.join(DATA_DIR, i))





# *********Numero de Amostras por classe - Agua potavel = 1 e não potável =0
vl_potavel = len(dataf.loc[dataf['Potability'] == 1]) 
#classe controle
vl_naopotavel = len(dataf.loc[dataf['Potability'] == 0])

# ******************* Dados Faltantes e valores = 0 **************************************************
dt_feature = dataf.iloc[:,:-1]
dt_target = dataf.iloc[:, -1]


#complementando dados faltantes com a média da coluna (coluna por coluna)
#dt_feature['ph'] = dt_feature['ph'].mask(dt_feature['ph'] == 0).fillna(dt_feature['ph'].mean())
dt_feature = dt_feature.mask(dt_feature == 0).fillna(dt_feature.mean())

print(dt_feature.head(10))



#print(dt_target.head(10))

####determinando a Média Aritimética, moda, mediana do, variancia e desvio padrao das features.

def dados_estatisticos(nome_feature):
    print("Dados da minha Feature ",nome_feature)

    print("Mediana: ", statistics.median(dt_feature[nome_feature]))
    print("Menor valor: ", min(dt_feature[nome_feature]))
    print("Maior valor PH: ", max(dt_feature[nome_feature]))
    print("Media Aritimética ", statistics.mean(dt_feature[nome_feature]))
    print("Desvio Padrão: ", statistics.stdev(dt_feature[nome_feature]))
    print("Variancia: ", statistics.variance(dt_feature[nome_feature]))

    

def analise_dispersao_dados():
    vfeatures = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
    valores = [statistics.stdev(dt_feature['ph']), statistics.stdev(dt_feature['Hardness']), statistics.stdev(dt_feature['Solids'])
    , statistics.stdev(dt_feature['Chloramines']), statistics.stdev(dt_feature['Sulfate']), statistics.stdev(dt_feature['Conductivity']),
    statistics.stdev(dt_feature['Organic_carbon']),statistics.stdev(dt_feature['Trihalomethanes']),statistics.stdev(dt_feature['Turbidity'])]


    plt.bar(vfeatures, valores, color='blue')
    plt.title('Gráfico que relaciona as features e suas dispersões de Dados')
    plt.xlabel('Feature')
    plt.ylabel('Valor do desvio Padrão')
    plt.show()
    #Ao analisar esse gráfico, notamos um valor de desvio extremamente alto(discrepante) em minha feature 'Solids' em rela
    #ção a minhas outras features



   


def informacoes_basicas():
    # Apresentando informações das classes
    print(dataf['Potability'].value_counts())
    print('\n Valores faltantes: \n', dataf.isnull().sum())
    print('Colunas com valores = 0: \n',(dataf==0).sum())
    print("O conjunto de dados possui %d linhas e %d colunas"%(len(dataf[:]), len(dataf.columns)))
    print('  %d da classe potável, que correspondem a %.2f%% dos dados' %(vl_potavel, vl_potavel /(vl_potavel+vl_naopotavel)*100))
    print('  %d da classe não potável, que correspondem a %.2f%% dos dados' %(vl_naopotavel, vl_naopotavel /(vl_potavel+vl_naopotavel)*100))








# verificando o Balanceamento dos dados das classes




def grafico_seaborn_classes():
    # Com Seaborn
    sns.factorplot('Potability', data=dataf, kind='count')
    plt.ylabel('Variaveis')
    plt.title("Balanceamento das classes")
    plt.show()




##grafico de relacao#############################




def correlacao():
    corr = dataf.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.title("Análise de correlação")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
   # plt.subplots(figsize=(20, 20))
    plt.show()
    
# Boxplot
def boxplot_todos():
    #Observamos uma grande diferenca de Valores da feature Solids das demais
    f, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor('#120A8F')
    ax.set(xlim=(-0.5, 1000))
    plt.ylabel('Variaveis')
    plt.title("Overview Dataset")
    ax = sns.boxplot(data=dataf, orient='v', palette='Set2')
    plt.show()


def boxplot_sem_feature_Solids():
    f, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor('#120A8F')
    ax.set(xlim=(-0.5, 1000))
    plt.ylabel('Variaveis')
    plt.title("Overview Dataset")
    ax = sns.boxplot(data=dataf.drop(columns=['Solids']), orient='v', palette='Set2')
    plt.show()


def boxplot_feature_solids():
    f, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor('#120A8F')
    ax.set(xlim=(-0.5, 1000))
    plt.ylabel('Variaveis')
    plt.title("Overview Dataset")
    ax = sns.boxplot(data=dataf['Solids'], orient='v', palette='Set2')
    plt.show()

# ******************* Preparando os modelos de ML **************************************************

# Criar uma lista de armazenamento de accuracia
accuracy_PC = []
accuracy_NB = []
accuracy_KNN = []
accuracy_LR = []
accuracy_SVM = []
accuracy_RF = []
accuracy_TD = []

def exibir_relacao_algoritmos_ML(media_perc,media_nb,media_knn,media_svn,media_tree):
    metodo = ["Perceptron", "Naive Bayes", "KNN k=3", "SVM", "Decision Tree"]
    lista = [media_perc,media_nb,media_knn,media_svn,media_tree]

    plt.bar(metodo, lista, color='blue')
    plt.title('Gráfico de comparação ML pela acurácia média')
    plt.xlabel('Algoritmo Aplicado')
    plt.ylabel('Acurácia')
    plt.show()



def Relacao_algoritmos_ML():
    print('\n ************************* Resultados ************************ \n')
    rould = 0.10
    epochs = 1  



    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.30, random_state=i)



        print('Divisão dos dados ------------------ Iteração N°',i+1,'\n')
        print('X_train: ',len(X_train))
        print('X_test: ',len(X_test))
        print('y_train: ',len(y_train))
        print('y_test: ',len(y_test))

        amostras_c0 = len(y_train.loc[y_train == 0])
        amostras_c1 = len(y_train.loc[y_train == 1])
        relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
        relacao_amostrasc1= (amostras_c1/len(y_train))*100

        print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
        print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
        print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
        print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% \n" % relacao_amostrasc1)



        # Perceptron
        percep = Perceptron()
        percep.fit(X_train, y_train)
        percep.predictions = percep.predict(X_test)
        acc_perpep = percep.score(X_test, y_test)

        # Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb.predictions = gnb.predict(X_test)
        acc_nb = gnb.score(X_test, y_test)

        # KNN com 3 vizinhos
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        knn.predictions = knn.predict(X_test)
        acc_knn = knn.score(X_test, y_test)
         

    

        # Suport Vector Machine
        vc = svm.SVC()
        vc.fit(X_train, y_train)
        vc.predictions = vc.predict(X_test)
        acc_svc = vc.score(X_test, y_test)


        # Arvore de decisão largura maxima = 3
        tree_decision =  DecisionTreeClassifier(
                           criterion='entropy',
                           max_depth=5)

        tree_decision.fit(X_train, y_train)
        tree_decision.predictions = tree_decision.predict(X_test)
        acc_tree = tree_decision.score(X_test, y_test)
        
        


        # Accuracy
        accuracy_PC.append(acc_perpep)
        accuracy_NB.append(acc_nb)
        accuracy_KNN.append(acc_knn)
        accuracy_SVM.append(acc_svc)

        accuracy_TD.append(acc_tree)

        print('Acc Perceptron: ', acc_perpep)
        print('Acc Bayes: ', acc_nb)
        print('Acc KNN: ', acc_knn)
        print('Acc Suport Vector Machine: ', acc_svc)
        print('Acc Arvore de Decisão: ', acc_tree)
 
       
   # print('Vetor de Acc Perceptron: ', accuracy_PC)
    #print('Vetor de Acc Bayes: ', accuracy_NB)
    #print('Vetor de Acc KNN: ', accuracy_KNN)
    #print('Vetor de Acc Suport Vector Machine: ', accuracy_SVM)
    #print('Vetor de Acc Arvore de Decisão: ', accuracy_TD)
 
    media_perc = (statistics.mean(accuracy_PC)*100)
    media_nb = (statistics.mean(accuracy_NB)*100) 
    media_knn = (statistics.mean(accuracy_KNN)*100)
    media_svn = (statistics.mean(accuracy_SVM)*100)
    media_tree = (statistics.mean(accuracy_TD)*100)


    print('Acurácia média Perceptron: %.2f%%'% media_perc)
    print('Acurácia média Naive Bayes: %.2f%%'% media_nb)
    print('Acurácia média KNN: %.2f%%'% media_knn)
    print('Acurácia média Suport Vector Machine: %.2f%%'% media_svn)
    print('Acurácia média Árvore de decisão: %.2f%%'% media_tree)
    print('\n')
    print('\n')
    exibir_relacao_algoritmos_ML(media_perc,media_nb,media_knn,media_svn,media_tree)

def plot_confusion_matrix(y_true, result,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, result)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, result)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def analise_KNN():
     X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3)

     print('Divisão dos dados\n')
     print('X_train: ',len(X_train))
     print('X_test: ',len(X_test))
     print('y_train: ',len(y_train))
     print('y_test: ',len(y_test))

     amostras_c0 = len(y_train.loc[y_train == 0])
     amostras_c1 = len(y_train.loc[y_train == 1])
     relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
     relacao_amostrasc1= (amostras_c1/len(y_train))*100

     print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
     print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
     print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
     print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% " % relacao_amostrasc1)
     #Definindo o número de vizinhos
     #obtendo o melhor valor de k
     

     knn = KNeighborsClassifier(n_neighbors=3)
     knn.fit(X_train, y_train)
     knn.predictions = knn.predict(X_test)
     acc_knn = knn.score(X_test, y_test)
     print("Acurácia calculada previamente com o valor de k=3",acc_knn)


     #verificando a taxa de acerto para cada classe.
     print("Imprimindo a matriz com as métricas de classificação")
     print(metrics.classification_report(y_test, knn.predictions))

     np.set_printoptions(precision=2)
     #imprimindo a matriz de confusão 
     plot_confusion_matrix(y_test, knn.predictions, normalize= True,title='Normalized confusion matrix')
     plt.show()

      #####Determinando o melhor valor de K##########################

     #recebendo a lista com valores de K de 1 a 30
     k_list = list(range(1,31))
     k_values = dict(n_neighbors = k_list)
     #instanciando um GridSearch
     grid = GridSearchCV(knn, k_values, cv=5, scoring='accuracy')
     #treinando o objeto
     grid.fit(X_train,y_train)

     # Apresentando a melhor acurácia com o melhor valor de k
     print('O valor de k otimizado é = {} com valor {} de acurácia'.format(grid.best_params_, grid.best_score_))


def analise_Perceptron():
     X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3)

     print('Divisão dos dados\n')
     print('X_train: ',len(X_train))
     print('X_test: ',len(X_test))
     print('y_train: ',len(y_train))
     print('y_test: ',len(y_test))

     amostras_c0 = len(y_train.loc[y_train == 0])
     amostras_c1 = len(y_train.loc[y_train == 1])
     relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
     relacao_amostrasc1= (amostras_c1/len(y_train))*100

     print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
     print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
     print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
     print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% " % relacao_amostrasc1)
    

     percep = Perceptron()
     percep.fit(X_train, y_train)
     percep.predictions = percep.predict(X_test)

   

     #verificando a taxa de acerto para cada classe.
     print(metrics.classification_report(y_test, percep.predictions))
     
     np.set_printoptions(precision=2)

     #imprimindo a matriz de confusão  
     plot_confusion_matrix(y_test, percep.predictions, normalize= True,title='Normalized confusion matrix')
     plt.show()

def analise_Naive_Bayes():
     X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3)

     print('Divisão dos dados\n')
     print('X_train: ',len(X_train))
     print('X_test: ',len(X_test))
     print('y_train: ',len(y_train))
     print('y_test: ',len(y_test))

     amostras_c0 = len(y_train.loc[y_train == 0])
     amostras_c1 = len(y_train.loc[y_train == 1])
     relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
     relacao_amostrasc1= (amostras_c1/len(y_train))*100

     print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
     print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
     print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
     print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% " % relacao_amostrasc1)
    
     gnb = GaussianNB()
     gnb.fit(X_train, y_train)
     gnb.predictions = gnb.predict(X_test)

   

     #verificando a taxa de acerto para cada classe.
     print(metrics.classification_report(y_test, gnb.predictions))
     
     np.set_printoptions(precision=2)
     #imprimindo a matriz de confusão
     plot_confusion_matrix(y_test, gnb.predictions, normalize= True,title='Normalized confusion matrix')
     plt.show()



def analise_Arvore_Decisao():
     X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3)

     print('Divisão dos dados\n')
     print('X_train: ',len(X_train))
     print('X_test: ',len(X_test))
     print('y_train: ',len(y_train))
     print('y_test: ',len(y_test))

     amostras_c0 = len(y_train.loc[y_train == 0])
     amostras_c1 = len(y_train.loc[y_train == 1])
     relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
     relacao_amostrasc1= (amostras_c1/len(y_train))*100

     print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
     print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
     print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
     print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% " % relacao_amostrasc1)
   

     tree_decision =  DecisionTreeClassifier(
                           criterion='entropy',
                           max_depth=3)

     tree_decision.fit(X_train, y_train)
     tree_decision.predictions = tree_decision.predict(X_test)

   

     #verificando a taxa de acerto para cada classe.

     print(metrics.classification_report(y_test, tree_decision.predictions))
     
     np.set_printoptions(precision=2)
     #imprimindo a matriz de confusão
    
     plot_confusion_matrix(y_test, tree_decision.predictions, normalize= True,title='Normalized confusion matrix')
     plt.show()


def analise_SVM():
     X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3)

     print('Divisão dos dados\n')
     print('X_train: ',len(X_train))
     print('X_test: ',len(X_test))
     print('y_train: ',len(y_train))
     print('y_test: ',len(y_test))

     amostras_c0 = len(y_train.loc[y_train == 0])
     amostras_c1 = len(y_train.loc[y_train == 1])
     relacao_amostrasc0 = (amostras_c0/(len(y_train)))*100
     relacao_amostrasc1= (amostras_c1/len(y_train))*100

     print('Quantidade de amostras da classe 0 (Não potável): ',amostras_c0 )
     print('Relação entre amostras da classe 0 e o total de amostras: %.2f%% ' % relacao_amostrasc0)
     print('Quantidade de amostras da classe 1 (Potável): ', amostras_c1)
     print("Relação entre amostras da classe 1 e o total de amostras: %.2f%% " % relacao_amostrasc1)
   

     vm =  svm.SVC()

     vm.fit(X_train, y_train)
     vm.predictions = vm.predict(X_test)

   

     #verificando a taxa de acerto para cada classe.

   #  print(metrics.classification_report(y_test, vm.predictions))
     
     np.set_printoptions(precision=2)
     #imprimindo a matriz de confusão
    
     plot_confusion_matrix(y_test, vm.predictions, normalize= True,title='Normalized confusion matrix')
     plt.show()







#Relacao_algoritmos_ML()
#analise_Perceptron()
#boxplot_todos()
#correlacao()
#analise_KNN()
#analise_Perceptron()
#analise_Naive_Bayes()
#analise_SVM()#################
#analise_Arvore_Decisao()
#dados_estatisticos('ph')
#dados_estatisticos('Chloramines')
#dados_estatisticos('Solids')
#dados_estatisticos('Conductivity')
#informacoes_basicas()
grafico_seaborn_classes()
#analise_dispersao_dados()
#boxplot_sem_feature_Solids()
#boxplot_feature_solids()