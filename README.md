# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: start the program

Step 2: Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

Step 3: Split the data into training and test sets using train_test_split.

Step 4: Create and fit a logistic regression model to the training data.

Step 5: Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

Step 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

Step 7:End the program. 

## Program:
```
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("Placement_Data_Full_Class (1).csv")
df.head()
```
## Output:
![image](https://github.com/user-attachments/assets/eb481be1-fb5b-44cb-8e8a-e56d1565f2b7)
```
df.tail()

```
## Output:

![image](https://github.com/user-attachments/assets/c7a79c9d-702a-4fb8-92cc-1140f14d728b)
```
df.drop('sl_no',axis=1)
```
## Output:
![image](https://github.com/user-attachments/assets/085673ac-08b9-44ca-9dec-08293fad8f15)
```
df.drop('sl_no',axis=1,inplace=True)
```
```
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df.dtypes

```
## Output:
![image](https://github.com/user-attachments/assets/fbffd6a9-161b-4c2e-ae47-e2de55f9e58d)
```
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes 

```
```
df.info()
```
```
df.head()


```
##Output:
![image](https://github.com/user-attachments/assets/5242d126-61b1-4861-a39d-4977bc414e07)

```
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

```
```
Y
```
## Output:
![image](https://github.com/user-attachments/assets/635b1f4b-716a-47de-9dfe-5859845d9321)
```
X
```
## Output:
![image](https://github.com/user-attachments/assets/c901dc0f-dc10-4b2c-8da1-67952864793d)
```
X.shape
```
## output:

![image](https://github.com/user-attachments/assets/651fb1e2-72a6-4855-af59-b0049eeec357)
```
Y.shape
```
## Output:
![image](https://github.com/user-attachments/assets/c645f546-2198-44ef-86b3-0f733d5467c5)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
```
```
X_train.shape
```
## Output:
![image](https://github.com/user-attachments/assets/3fbbb773-3a2c-4652-abee-4e5be65bc40d)
```
Y_train.shape
```
## Output:
![image](https://github.com/user-attachments/assets/aac968e8-839d-453c-946a-4a6a9121f6ae)
```
Y_test.shape
```
## Output:
![image](https://github.com/user-attachments/assets/21900a5a-c8df-4e2d-a217-dafbfc220f7f)
```
clf = LogisticRegression()
clf.fit(X_train,Y_train)
```
```
Y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
```
```
cf = confusion_matrixaccuracy_score
print(cf)
```
## Output:
![image](https://github.com/user-attachments/assets/bd2bd303-5413-4bd5-9604-a3c77f9989b7)
```
accuracy=accuracy_score(Y_test,Y_pred)
```
```
print(accuracy)
```
## Output:
![image](https://github.com/user-attachments/assets/e284eb22-ce9a-44b0-a4a1-12b90505a45a)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
