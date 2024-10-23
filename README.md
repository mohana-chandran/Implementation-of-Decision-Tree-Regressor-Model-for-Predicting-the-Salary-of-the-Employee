# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mohanachandran.J.B
RegisterNumber:  212221080049
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
![Screenshot 2024-10-22 233310](https://github.com/user-attachments/assets/fa22fd88-5fa6-4141-81a4-9c3c6956dd4f)

![Screenshot 2024-10-22 233321](https://github.com/user-attachments/assets/342c746e-4e50-4738-a143-25e130c185c1)

![Screenshot 2024-10-22 233327](https://github.com/user-attachments/assets/a47d08b4-b5ea-485f-9bef-42558227fe73)
### mse:
![Screenshot 2024-10-22 233338](https://github.com/user-attachments/assets/f84fd1cc-b99f-44f6-8b90-0f2955edc97d)
### r2:
![Screenshot 2024-10-22 233348](https://github.com/user-attachments/assets/d61c8dfe-e6bf-4cdc-81d0-74e6906afa89)
![Screenshot 2024-10-22 233413](https://github.com/user-attachments/assets/0b951925-21fc-407b-b261-65002c8a1720)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
