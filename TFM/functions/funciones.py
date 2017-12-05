
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve,auc,accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from matplotlib.font_manager import FontProperties
from sklearn.feature_selection import RFECV
from sklearn import preprocessing


# In[2]:

def EliminaNulos(dataSet,eje):
    if(eje==0):
        dataSetOut=dataSet.dropna(thresh=len(dataSet.columns) / 2,axis=eje)
    else:
        dataSetOut=dataSet.dropna(thresh=len(dataSet) / 2,axis=eje)
    return dataSetOut


# In[3]:

def eliminaOutliers(dataSet):
    dataSet[dataSet.apply(lambda x: np.abs(x - x.mean())/x.std() < 3).all(axis=1)]
    return dataSet


# In[4]:

def NormalizaDatos(dataSet,columnas,tipo):
    outDataSet=dataSet.copy()
    if tipo=="MinMax":
        scala = preprocessing.MinMaxScaler().fit(outDataSet[columnas])
    else:
        scala = preprocessing.StandardScaler().fit(outDataSet[columnas])
    outDataSet[columnas] = scala.transform(outDataSet[columnas])
    return outDataSet


# In[5]:

def BoxPlotVariable(datos, variable):
    
    datos.boxplot(column=variable)
    plt.title("Boxplot de la variable '"+variable+"'")
    plt.show()


# In[6]:

def BoxPlotDataset(datos,columnas):

    for column in columnas:
        
        BoxPlotVariable(datos,column)


# In[7]:

def BarPlotVariable(dataSet,variable):
    
    valoresVariable=dataSet[variable].unique()
    valoresVariable=map(int, valoresVariable)
    valoresVariable.sort()
    fig, ax = plt.subplots() 
    index=np.arange(valoresVariable[0],valoresVariable[len(valoresVariable)-1]+1)
    anchura = 0.35
    anchuraBarra = 0.35
    opacity = 0.5
    valoresDefault=[0]*len(index)
    valoresNoDefault=[0]*len(index)
    for i in xrange(len(valoresVariable)):
        valoresDefault[i]=len(dataSet[(dataSet[variable]==valoresVariable[i]) & (dataSet['default payment next month']==1)])
        valoresNoDefault[i]=len(dataSet[(dataSet[variable]==valoresVariable[i]) & (dataSet['default payment next month']!=1)])
    plt.bar(index, valoresDefault,anchuraBarra,alpha=opacity,color='r',label='Default')
    plt.bar(index+anchuraBarra, valoresNoDefault, anchuraBarra,alpha=opacity,color='b',label='No default')
    plt.title("Diagrama de barras de la variable '"+variable+"'")
    plt.xticks(index + anchuraBarra, map(str, valoresVariable) )
    plt.legend()
    plt.tight_layout()


# In[8]:

def BarPlotVariablesDataset(dataSet,columnas):
    for column in columnas:
        BarPlotVariable(dataSet,column)
    plt.show()


# In[9]:

def PreparaDatosEntrenamientoTest(dataSet):
    msk = np.random.rand(len(dataSet)) < 0.8
    train=dataSet[msk]
    test=dataSet[~msk]
    trainx = train.ix[:,:(len(dataSet.columns)-1)]
    trainy = train.ix[:,(len(dataSet.columns)-1):]
    testx = test.ix[:,:(len(dataSet.columns)-1)]
    testy = test.ix[:,(len(dataSet.columns)-1):]
    return trainx,trainy,testx,testy


# In[10]:

def Clasifica(clf,trainX,trainY,testX):
    clf.fit(trainX,trainY)
    prediccion=clf.predict(testX)
    return prediccion


# In[11]:

def CurvaRocAnalisis(standartTarget,predictTarget):
    print("Precision:",accuracy_score(standartTarget,predictTarget))
    print("F1Score:",f1_score(standartTarget,predictTarget))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(standartTarget,predictTarget)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[12]:

from sklearn.model_selection import StratifiedKFold
from scipy import interp
from matplotlib.font_manager import FontProperties

def CrossValidation (dataSet,clf,inicio,fin):
    dataSetx=dataSet.ix[:,:(len(dataSet.columns)-1)]
    dataSety =dataSet['default payment next month']
    aucMedioMax=0
    for i in range(inicio,fin):
            print(i)
            cv = StratifiedKFold(n_splits=i)
            truePositiveRateArray = []
            aucArray = []
            meanAuc=0
            standardAuc=0
            falsePositiveRateMedio = np.linspace(0, 1, 100)
            j = 0
            for train, test in cv.split(dataSetx,dataSety):
                    prediccion = clf.fit(dataSetx.ix[train], dataSety.ix[train]).predict_proba(dataSetx.ix[test])
                    # Compute ROC curve and area the curve
                    falsePositiveRate, truePositiveRate, thresholds = roc_curve(dataSety[test], prediccion[:, 1])
                    truePositiveRateArray.append(interp(falsePositiveRateMedio, falsePositiveRate, truePositiveRate))
                    truePositiveRateArray[-1][0] = 0.0
                    roc_auc = auc(falsePositiveRate, truePositiveRate)
                    aucArray.append(roc_auc)
                    plt.plot(falsePositiveRate, truePositiveRate, lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.4f)' % (j, roc_auc))
                    j += 1
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Aleatorio', alpha=.8)
            truePositiveRateMedio = np.mean(truePositiveRateArray, axis=0)
            truePositiveRateMedio[-1] = 1.0
            aucMedio = np.mean(aucArray,axis=0)
            aucStandard = np.std(aucArray,axis=0)
            plt.plot(falsePositiveRateMedio, truePositiveRateMedio, color='b',
                     label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (aucMedio, aucStandard),
                     lw=2, alpha=.8)

            truePositiveRateStandard = np.std(truePositiveRateArray, axis=0)
            tprs_upper = np.minimum(truePositiveRateMedio + truePositiveRateStandard, 1)
            tprs_lower = np.maximum(truePositiveRateStandard + truePositiveRateStandard, 0)
            plt.fill_between(falsePositiveRateMedio, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            fontP = FontProperties()
            fontP.set_size('small')
            plt.legend(loc=6, bbox_to_anchor=(0.5,0))
            #plt.legend(prop=fontP)
            plt.show()
            if (aucMedio>aucMedioMax):
                aucMedioMax=aucMedio
    return aucMedioMax


# In[13]:

def CalculaNumeroVariablesOptimas(clf,datasetx,y):
    
    selector=RFECV(clf,step=1,cv=10,scoring='roc_auc',verbose=2)
    
    selector=selector.fit(datasetx,y)
    
    return selector.support_,selector.ranking_,selector.n_features_

