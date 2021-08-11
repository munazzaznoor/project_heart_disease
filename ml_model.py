from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#import seaborn as sns
import numpy as np


#IMPORTING THE DATASET

label_csv=pd.read_csv("framingham.csv")
print(label_csv.describe())

print(label_csv.head())
type(label_csv)
#sns.pairplot(label_csv,hue="TenYearCHD")

#CHECKING WHETHER THE DATASET HAS MISSING VALUES OR NOT
label_csv.isnull().sum()

#DROP THE MISSING VALUES
new=label_csv.dropna()

new["TenYearCHD"].value_counts().plot.bar(figsize=(10,10))

labels=new["TenYearCHD"]

labels=np.array(labels)
print(labels)

print(type(new))


#CONVERTING THE DATA TO NUMPY ARRAY
mydata_array=np.array(new)
print(type(mydata_array))

y=new.TenYearCHD
x=new.drop("TenYearCHD",axis=1)
print(y)
x.head()


#SPLITTING THE DATASET INTO TEST AND TRAIN DATASETS
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print("shape of original dataset :", new.shape)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)

#DIMENSIONS
print(x_train.ndim)
print(y_train.ndim)
print(x_test.ndim)
print(y_test.ndim)


knn=KNeighborsClassifier(n_neighbors=1)
print(knn)

#FITTING THE ALGORITHM
knn.fit(x,y)

#N_VALUE=1
print(knn)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

knn.predict([[0,41,2,0,0,1,0,0,0,260,110,61.5,25.42,80,77]])


error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,y,cv=10)
    error_rate.append(1-score.mean())

#plt.figure(figsize=(10,30))

plt.plot(range(1,40),error_rate,color="red",linestyle="dashed",marker="o",markerfacecolor="red",markersize=10)
#plt.plot(range(1,40),accuracy_rate,color="blue",linestyle="dasshed",marker="o",markerfacecolor="red",markersize=10)
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

#N_VALUE = 10
knn=KNeighborsClassifier(n_neighbors=10)
print(knn)
knn_model=knn.fit(x,y)
knn.predict([[0,41,2,0,0,1,0,0,0,260,110,61.5,25.42,80,77]])

pred=knn_model.predict(x_test)
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
#MAKING PREDICTION ON CUSTOM_DATA
custom_data=np.array([[1,41,2,0,0,23,0,0,0,260,110,61.5,25.42,80,77]])
prediction=knn_model.predict(custom_data)
print(prediction)
sc=MinMaxScaler(feature_range=(0,1))
X_scaled=sc.fit_transform(x_train)

pickle.dump(knn_model,open("model_pickle.sav","wb"))

pickle.dump(sc,open("scaler.sav","wb"))
