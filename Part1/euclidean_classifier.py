import numpy as np

def euclidianDistance(x1,x2,d):
    if d==1:
        distance= np.sqrt(np.sum(np.square(x1 - x2)))
    else:
        distance=np.sqrt(np.sum((x1-x2).T.dot(x1-x2)))
    return distance



def euclidean_classifier(x, listofμ, listofΣ):
    print("X-sample:\n{0}".format(x))
    distance = np.empty([len(listofμ)])
    for i in np.arange(len(listofμ)):
        distance[i] = (x - listofμ[i]).dot((x - listofμ[i]).transpose())
        print("Euclidean distance for X-sample to class {0} = {1}".format(i + 1, distance[i]))
    indexClassContain = next(i for i, x in enumerate(distance) if x == min(distance))

    print("By minimum euclidean distance classifier belongs to >> class {0}  \n *********************** \n ".format(
        indexClassContain + 1, min(distance), x))

    return ""

def classification_error_Euclidian(test_data,μ,testsize):
    euclidianFalses=0
    for j in np.arange(len(μ)):
        for i in range(test_data[j].shape[0]):
            distanceto1=euclidianDistance(test_data[j][i],μ[0],3)
            distanceto2=euclidianDistance(test_data[j][i],μ[1],3)
            distanceto3=euclidianDistance(test_data[j][i],μ[2],3)
            if(j ==0):
                if distanceto1>distanceto2 or distanceto1>distanceto3:
                    euclidianFalses= euclidianFalses + 1
            if(j ==1):
                if distanceto2 > distanceto1 or distanceto2 > distanceto3:
                    euclidianFalses = euclidianFalses + 1
            if(j ==2):
                if distanceto3 > distanceto1 or distanceto3 > distanceto2:
                    euclidianFalses = euclidianFalses + 1
    euclidianError=euclidianFalses/testsize
    print("Euclidian Minimun Distance Classifier Error:",euclidianError," Using theoretical Mean and Covariance")