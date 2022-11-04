from Part1 import gauss_classes_generation
from Part1 import bayes_classifier
from Part1 import mahalanobis_classifier
from Part1 import euclidean_classifier

import matplotlib.pyplot as plt
import numpy as np
print("Task 6 part a")
trainsize=1000
d =2
p1X4=p2X4= p3X4=0.3
listofPX4=[p1X4,p2X4,p3X4]

p1X5=0.8
p2X5=0.1
p3X5=0.1
listofPX5=[p1X5,p2X5,p3X5]

μ1=np.array([1,1])
μ2=np.array([4,4])
μ3=np.array([8,1])
listofμ=[μ1,μ2,μ3]

Σ1=Σ2=Σ3=4*np.identity(2)
listofΣ=[Σ1,Σ2,Σ3]

dataset_X4=gauss_classes_generation.createMultivariateClassData(3,trainsize,listofPX4,listofμ,listofΣ)
dataset_X5=gauss_classes_generation.createMultivariateClassData(3,trainsize,listofPX5,listofμ,listofΣ)

x =dataset_X4[1][50]
print("sample of dataset X4:",dataset_X4[0][1],dataset_X4[0][2],"....",dataset_X4[0][5])
print("sample of dataset X5:",dataset_X5[0][1],dataset_X5[0][2],"....",dataset_X5[0][5])

plt.figure(1)
plt.plot(dataset_X4[0][:,0], dataset_X4[0][:,1], 'r.')
plt.plot(dataset_X4[1][:,0], dataset_X4[1][:,1], 'b.')
plt.plot(dataset_X4[2][:,0], dataset_X4[2][:,1], 'y.')
plt.xlim(-25, 25)
plt.ylim(-8, 8)
plt.grid()
plt.show()

plt.figure(2)
plt.plot(dataset_X5[0][:,0], dataset_X5[0][:,1], 'r.')
plt.plot(dataset_X5[1][:,0], dataset_X5[1][:,1], 'b.')
plt.plot(dataset_X5[2][:,0], dataset_X5[2][:,1], 'y.')
plt.xlim(-25, 25)
plt.ylim(-8, 8)
plt.grid()
plt.show()

print("Task 6 part b")

X4_Sample =dataset_X4[0][50]
X5_Sample =dataset_X5[0][50]

# # Try classifier
# dataset X4
print("For dataset X4 the classifier: ")
bayes_classifier.bayes_classifier(X4_Sample,listofμ,listofΣ,listofPX4,d)
euclidean_classifier.euclidean_classifier(X4_Sample,listofμ,listofΣ)
mahalanobis_classifier.mahalanobis_classifier(X4_Sample,listofμ,listofΣ)
bayes_classifier.classification_error_Bayesian(dataset_X4,listofμ,listofΣ,listofPX4,trainsize)
euclidean_classifier.classification_error_Euclidian(dataset_X4,listofμ,trainsize)
mahalanobis_classifier.classification_error_mahalanobis(dataset_X4,listofμ,listofΣ,trainsize)

# dataset X5
print("\nــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ\n","For dataset X5 the classifier: ")
bayes_classifier.bayes_classifier(X5_Sample,listofμ,listofΣ,listofPX5,d)
euclidean_classifier.euclidean_classifier(X5_Sample,listofμ,listofΣ)
mahalanobis_classifier.mahalanobis_classifier(X5_Sample,listofμ,listofΣ)
bayes_classifier.classification_error_Bayesian(dataset_X5,listofμ,listofΣ,listofPX5,trainsize)
euclidean_classifier.classification_error_Euclidian(dataset_X5,listofμ,trainsize)
mahalanobis_classifier.classification_error_mahalanobis(dataset_X5,listofμ,listofΣ,trainsize)