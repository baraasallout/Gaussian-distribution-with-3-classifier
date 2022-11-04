from Part1 import gauss_classes_generation
import matplotlib.pyplot as plt
import numpy as np

print("Task 1 part a")
trainsize=1000
d =2
p1=p2=p3=0.3
listofP=[p1,p2,p3]

μ1=np.array([1,1])
μ2=np.array([7,7])
μ3=np.array([15,1])
listofμ=[μ1,μ2,μ3]

Σ1=np.array([[12,0],[0,1]])
Σ2=np.array([[8,3],[3,2]])
Σ3=np.array([[2,0],[0,2]])

listofΣ=[Σ1,Σ2,Σ3]

test_data=gauss_classes_generation.createMultivariateClassData(3,trainsize,listofP,listofμ,listofΣ)

x =test_data[1][15]
print("sample of dataset:",test_data[0][1],test_data[0][2],"....",test_data[0][5])
plt.figure(1)
plt.plot(test_data[0][:,0], test_data[0][:,1], 'r.')
plt.plot(test_data[1][:,0], test_data[1][:,1], 'b.')
plt.plot(test_data[2][:,0], test_data[2][:,1], 'y.')
plt.axis('equal')
plt.xlim(-25, 25)
plt.ylim(-15, 15)
plt.grid()
plt.show()


print("Task 1 part b")
trainsize=1000
d =2
p1=0.6
p2=0.3
p3=0.1
listofP=[p1,p2,p3]

μ1=np.array([1,1])
μ2=np.array([7,7])
μ3=np.array([15,1])
listofμ=[μ1,μ2,μ3]

Σ1=np.array([[12,0],[0,1]])
Σ2=np.array([[8,3],[3,2]])
Σ3=np.array([[2,0],[0,2]])

listofΣ=[Σ1,Σ2,Σ3]

test_data=gauss_classes_generation.createMultivariateClassData(3,trainsize,listofP,listofμ,listofΣ)

x =test_data[1][15]
print("sample of dataset:",test_data[0][1],test_data[0][2],"....",test_data[0][5])
plt.figure(2)
plt.plot(test_data[0][:,0], test_data[0][:,1], 'r.')
plt.plot(test_data[1][:,0], test_data[1][:,1], 'b.')
plt.plot(test_data[2][:,0], test_data[2][:,1], 'y.')
plt.axis('equal')
plt.xlim(-25, 25)
plt.ylim(-15, 15)
plt.grid()
plt.show()

# # Try classifier
# bayes_classifier(x,listofμ,listofΣ,listofP,d)
# euclidean_classifier(x,listofμ,listofΣ)
# mahalanobis_classifier(x,listofμ,listofΣ)