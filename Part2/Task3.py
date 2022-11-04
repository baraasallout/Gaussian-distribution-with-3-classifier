from Part1 import gauss_classes_generation
from Part1 import bayes_classifier
from Part1 import mahalanobis_classifier
from Part1 import euclidean_classifier

import matplotlib.pyplot as plt
import numpy as np

print("Task 3 part a")
trainsize=1000
d =2
p1=p2=p3=0.3
listofP=[p1,p2,p3]

μ1=np.array([1,1])
μ2=np.array([14,7])
μ3=np.array([16,1])
listofμ=[μ1,μ2,μ3]

Σ1=Σ2=Σ3=np.array([[5,3],[3,5]])
listofΣ=[Σ1,Σ2,Σ3]
test_data=gauss_classes_generation.createMultivariateClassData(3,trainsize,listofP,listofμ,listofΣ)

x =test_data[1][50]
print("sample of dataset X2:",test_data[0][1],test_data[0][2],"....",test_data[0][5])

plt.plot(test_data[0][:,0], test_data[0][:,1], 'r.')
plt.plot(test_data[1][:,0], test_data[1][:,1], 'b.')
plt.plot(test_data[2][:,0], test_data[2][:,1], 'y.')
plt.axis('equal')
plt.xlim(-25, 25)
plt.ylim(-15, 15)
plt.grid()
plt.show()

# # Try classifier

print("\n Task 3 part b")

bayes_classifier.bayes_classifier(x,listofμ,listofΣ,listofP,d)
euclidean_classifier.euclidean_classifier(x,listofμ,listofΣ)
mahalanobis_classifier.mahalanobis_classifier(x,listofμ,listofΣ)
bayes_classifier.classification_error_Bayesian(test_data,listofμ,listofΣ,listofP,trainsize)
euclidean_classifier.classification_error_Euclidian(test_data,listofμ,trainsize)
mahalanobis_classifier.classification_error_mahalanobis(test_data,listofμ,listofΣ,trainsize)