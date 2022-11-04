import numpy as np
import math
def createMultivariateClassData(numberOfClasses,datasize,p,listofμ,listofΣ):
    listOfData=[]
    numberOfSamplesCreated=0 #check how many we have created so last class rounds to datasize!!!!
    for i in range(numberOfClasses-1):
        classSize=math.floor(datasize*p[i])
        listOfData.append(np.random.multivariate_normal(listofμ[i],listofΣ[i],size=classSize))
        numberOfSamplesCreated=numberOfSamplesCreated+classSize
    #for last class we need to take the remaining samples cause for sum of p we might not have round number of data
    lastclassSize=datasize-numberOfSamplesCreated
    listOfData.append(np.random.multivariate_normal(listofμ[-1],listofΣ[-1],size=lastclassSize))
    return listOfData