# Implementation-of-SVM-For-Spam-Mail-Detection
## Name: Renick Fabian Rajesh
## Reg No: 212224230227
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Renick Fabian Rajesh
RegisterNumber: 212224230227
*/
```
```
import numpy as np
import chardet
file = "spam.csv"
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.isnull().sum()
data.info()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
x_train
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
<img width="790" height="51" alt="image" src="https://github.com/user-attachments/assets/07885f68-0ddb-418d-be14-5c0935af3c2b" />


<img width="744" height="215" alt="image" src="https://github.com/user-attachments/assets/f34e3a61-38a4-41ae-9475-bdff6c093c32" />


<img width="201" height="146" alt="image" src="https://github.com/user-attachments/assets/0426c74c-695c-41dd-8b7c-002123aca6a3" />


<img width="408" height="272" alt="image" src="https://github.com/user-attachments/assets/1134a4a6-ec14-4f3f-8c31-49c4e3a1ac2f" />


<img width="1235" height="204" alt="image" src="https://github.com/user-attachments/assets/12fcbc3f-fc6f-4bf0-bf58-7300e995f479" />


<img width="1254" height="247" alt="image" src="https://github.com/user-attachments/assets/04f988b8-05aa-40e8-9bdf-0ffe10e5b140" />


<img width="670" height="64" alt="image" src="https://github.com/user-attachments/assets/ec3292af-1b34-4dbe-9663-702c77a8cd68" />


<img width="696" height="72" alt="image" src="https://github.com/user-attachments/assets/a7d4ef28-4cdd-4765-afbc-f0234e722d1b" />


<img width="675" height="42" alt="image" src="https://github.com/user-attachments/assets/439f6c2f-e3b1-4918-97a7-9e8191e030a7" />


<img width="211" height="56" alt="image" src="https://github.com/user-attachments/assets/cbded107-6657-4f58-a6ca-ed7dc714cd34" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
