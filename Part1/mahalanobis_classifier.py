import numpy as np
import math

def mahalanobisDistance(x,m,s,d):
    c=x-m
    if d>1:
        invS=np.linalg.inv(s)
        distance = np.sqrt(c.T.dot(invS).dot(c))
    else:
        invS = s ** -1
        distance = math.sqrt(c*invS*c)
    return distance

def mahalanobis_classifier(x, listofμ, listofΣ):
    distance = np.empty([len(listofμ)])
    print("X-sample:\n{0}".format(x))
    for i in np.arange(len(listofμ)):
        distance[i] = (x - listofμ[i]).dot(np.linalg.inv(listofΣ[i])).dot((x - listofμ[i]).transpose())
        print("Mahalanobis distance for X-sample to class {0} = {1}".format(i + 1, distance[i]))
    indexClassContain = next(i for i, x in enumerate(distance) if x == min(distance))

    print("By minimun mahalanobis distance classifier belongs to >> class {0}  \n *********************** \n ".format(
        indexClassContain + 1, min(distance), x))

    return ""

def classification_error_mahalanobis(test_data,μ,Σ,testsize):
    mahalanobisFalses=0
    for j in np.arange(len(μ)):
        for i in range(test_data[j].shape[0]):
            distanceto1=mahalanobisDistance(test_data[j][i],μ[0],Σ[0],3)
            distanceto2=mahalanobisDistance(test_data[j][i],μ[1],Σ[1],3)
            distanceto3=mahalanobisDistance(test_data[j][i],μ[2],Σ[2],3)
            if(j ==0):
                if distanceto1>distanceto2 or distanceto1>distanceto3:
                    mahalanobisFalses= mahalanobisFalses + 1
            if(j ==1):
                if distanceto2 > distanceto1 or distanceto2 > distanceto3:
                    mahalanobisFalses = mahalanobisFalses + 1
            if(j ==2):
                if distanceto3 > distanceto1 or distanceto3 > distanceto2:
                    mahalanobisFalses = mahalanobisFalses + 1
    mahalanobisFalses=mahalanobisFalses/testsize
    print("Mahalanobis Minimun Distance Classifier Error:",mahalanobisFalses," Using theoretical Mean and Covariance")