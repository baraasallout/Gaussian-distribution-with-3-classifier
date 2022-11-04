import numpy as np
import math

def classifyFunc(d,x,m,s,pw):
    c = x - m
    if d>1:
        detS=round(abs(np.linalg.det(s)),4)
        invS=np.linalg.inv(s)
        g = -0.5 * (c.T).dot(invS).dot(c) - (d / 2) * math.log(2 * math.pi) - 0.5 * math.log(detS) + math.log(pw)
    else:
        detS=abs(s)
        invS=s**-1
        g = -0.5 * c**2*invS- (d / 2) - (d / 2) * math.log(2 * math.pi) - 0.5 * math.log(detS) + math.log(pw)
    return g

def bayes_classifier(x,listofμ,listofΣ,pw ,d):
    print("X-sample:\n{0}".format(x))
    bayesclassify = np.empty([len(listofμ)])
    for i in np.arange(len(listofμ)):
        bayesclassify[i] = classifyFunc(d,x,listofμ[i],listofΣ[i],pw[i])
        print("Bayesian classification for X-sample to class {0} = {1}".format(i+1,bayesclassify[i]) )
    indexClassContain = next(i for i,x in enumerate(bayesclassify) if x == max(bayesclassify))

    print("By maximum Bayesian classification belongs to >> class {0}  \n *********************** \n ".format(indexClassContain+1,max(bayesclassify),x) )

    return ""


def classification_error_Bayesian(test_data,μ,Σ,p,testsize):
    bayesianFalses=0
    for j in np.arange(len(μ)):
        for i in range(test_data[j].shape[0]):
            distanceto1=classifyFunc(3,test_data[j][i],μ[0],Σ[0],p[0])
            distanceto2=classifyFunc(3,test_data[j][i],μ[1],Σ[1],p[1])
            distanceto3=classifyFunc(3,test_data[j][i],μ[2],Σ[2],p[2])
            if(j ==0):
                if distanceto1<distanceto2 or distanceto1<distanceto3:
                    bayesianFalses= bayesianFalses + 1
            if(j ==1):
                if distanceto2 < distanceto1 or distanceto2 < distanceto3:
                    bayesianFalses = bayesianFalses + 1
            if(j ==2):
                if distanceto3 < distanceto1 or distanceto3 < distanceto2:
                    bayesianFalses = bayesianFalses + 1
    bayesianError=bayesianFalses/testsize
    print("Bayesian Maximun Distance Classifier Error:",bayesianError," Using theoretical Mean and Covariance")