 import gdal
 import matplotlib.pyplot as plt
 import numpy as np # linear algebra
 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 from sklearn.model_selection import train_test_split
 from math import pi
 import seaborn as sns
 import sklearn.metrics as sm

 def MLestimator(dataset):
     mean = np.mean(dataset,axis=0)
     variance = np.var(dataset,axis=0)
     return(mean,variance)
 def variance_squaremat(var):
     return(np.multiply(var,np.identity(7)))
 def BayesClassifier(test_point,mean,variance):
     mean = np.array([mean])
     X = test_point-mean
     p1 = ((1/(2*pi)**0.5)/(np.linalg.det(variance)))**0.5
     p2 = np.exp(-0.5*np.matmul(np.matmul(X,np.linalg.inv(variance)),np.transpose(X))) 
     P = (1/3)*p1*p2 # 1/3 is the prior probability
     return(P)
 #loading data to different variables as 2D array
 img = gdal.Open(r'C:\Users\student\Downloads\Weather_Guj.tif')
 for i in range (1,9):
     T = img.GetRasterBand(i)
     if i==1:
         Tmin = T.ReadAsArray()
     if i==2:
         Tmax = T.ReadAsArray()
     if i==3:
         RH1 = T.ReadAsArray()
     if i==4:
         RH2 = T.ReadAsArray()
     if i==5:
         RF = T.ReadAsArray()
     if i==6:
         BSS = T.ReadAsArray()
     if i==7:
         WS = T.ReadAsArray()
     if i==8:
         AI = T.ReadAsArray() 

 #Prepare Features & Targets
 #creating 1D array of input data
 minT = np.transpose(np.reshape(Tmin,(1,np.product(Tmin.shape))))
 maxT = np.transpose(np.reshape(Tmax,(1,np.product(Tmax.shape))))
 rh1 = np.transpose(np.reshape(RH1,(1,np.product(RH1.shape))))
 rh2 = np.transpose(np.reshape(RH2,(1,np.product(RH2.shape))))
 rf = np.transpose(np.reshape(RF,(1,np.product(RF.shape))))
 bss = np.transpose(np.reshape(BSS,(1,np.product(BSS.shape))))
 ws = np.transpose(np.reshape(WS,(1,np.product(WS.shape))))
 ai = np.transpose(np.reshape(AI,(1,np.product(AI.shape))))


 #removing error pixel from class image
 for j in range (0,ai.shape[0]):
     if ai[j]<1:
         ai[j] = 0;

 #creating input array with all features
 x=np.concatenate((minT, maxT, rh1,rh2,rf,bss,ws,ai), axis=1)

 class0 = x[x[:,7]==0]
 class1 = x[x[:,7]==1]
 class2 = x[x[:,7]==2]
 class3 = x[x[:,7]==3]
 class4 = x[x[:,7]==4]

 m = 0.01

 train_c0, test_c0 = train_test_split(class0,test_size=m)
 train_c1, test_c1 = train_test_split(class1,test_size=m)
 train_c2, test_c2 = train_test_split(class2,test_size=m)
 train_c3, test_c3 = train_test_split(class3,test_size=m)
 train_c4, test_c4 = train_test_split(class4,test_size=m)


 test_data = np.concatenate((test_c0,test_c1,test_c2,test_c3,test_c4),axis = 0)


 mean_c0,variance_c0= MLestimator(train_c0[:,0:7])
 mean_c1,variance_c1= MLestimator(train_c1[:,0:7])
 mean_c2,variance_c2= MLestimator(train_c2[:,0:7])
 mean_c3,variance_c3= MLestimator(train_c3[:,0:7])
 mean_c4,variance_c4= MLestimator(train_c4[:,0:7])

 variance_c0 = variance_squaremat(variance_c0)
 variance_c1 = variance_squaremat(variance_c1)
 variance_c2 = variance_squaremat(variance_c2)
 variance_c3 = variance_squaremat(variance_c3)
 variance_c4 = variance_squaremat(variance_c4)



 count = 0
 n = np.shape(test_data)[0]
 Pred = np.zeros([n,1])
for i in range(n):
  s = test_data[i,0:7]
  t = test_data[i,7]

 # virginica flower type probability calculation call
 prob_c0 = BayesClassifier(s,mean_c0,variance_c0)

 prob_c1 = BayesClassifier(s,mean_c1,variance_c1)

 prob_c2 = BayesClassifier(s,mean_c2,variance_c2)

 prob_c3 = BayesClassifier(s,mean_c3,variance_c3)

 prob_c4 = BayesClassifier(s,mean_c4,variance_c4)


 if(max(prob_c0,prob_c1,prob_c2,prob_c3,prob_c4)==prob_c0):
     p = 0
     Pred[i,:] = p
 elif(max(prob_c0,prob_c1,prob_c2,prob_c3,prob_c4)==prob_c1):
     p = 1
     Pred[i,:] = p
 elif(max(prob_c0,prob_c1,prob_c2,prob_c3,prob_c4)==prob_c2):
     p = 2
     Pred[i,:] = p
 elif(max(prob_c0,prob_c1,prob_c2,prob_c3,prob_c4)==prob_c3):
     p = 3
     Pred[i,:] = p
 elif(max(prob_c0,prob_c1,prob_c2,prob_c3,prob_c4)==prob_c4):
     p = 4
     Pred[i,:] = p

 if(t==p):
     count+=1

print("The accuracy of the Bayes classifier which uses ML estimation is: %0.2f%%" %(count*100/n))
print(sm.classification_report(test_data[:,7], Pred))
print('MxL\n', sm.confusion_matrix(test_data[:,7], Pred))
print('MxL accuracy is',sm.accuracy_score(Pred,test_data[:,7]))

ax = sns.heatmap(sm.confusion_matrix(test_data[:,7], Pred), linewidth=0.1)
plt.show() 