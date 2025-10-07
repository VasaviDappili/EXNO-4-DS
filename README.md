# EXNO:4-Feature Scaling and Selection
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## Developed By: DAPPILI VASAVI
## Register no: 212223040030

## CODING AND OUTPUT:
```py
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/6f0f8b73-3335-4443-9ac8-aff7e5849c6b)

```py
data.isnull().sum()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/451a6149-5c70-4cae-9d53-c4894ae2e8e4)

```py
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/e909e09e-f12a-4114-8f4b-9de31056886e)

```py
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/80785ae2-4c93-43ee-904e-8afa82c8c839)

```py
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/c26296a0-3f88-4bea-bf79-02b5ae669153)

```py
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/d9a2e4db-be4b-47b2-8730-4cb7bffe0513)

```py
data2
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/57203963-f058-4216-9333-c4501452572c)

```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/cd218c4f-76f9-4cc9-9538-7b9eeec12b9d)
```py
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/f2a2209e-dd66-41b9-8610-3e75f516f6cb)
```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/d20c23e6-83a2-4190-9cf6-9b0f31be09d0)

```py
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/2e35eac6-5a50-443c-994e-f76f201a97fe)
```py
x=new_data[features].values
print(x)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/b3b040cd-c01e-41b1-b016-5a997c04d144)

```py

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/cb077c23-9c1e-4d18-9c88-467ce8803cbb)

```py
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/da211449-6ff6-4da2-af63-c0c42aa74a5a)
```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/a843759f-5e5c-41fa-af77-0da66b255bc4)
```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/be2b2b8a-a9c6-45fa-903b-5b81c1fae78f)
```py
data.shape
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/33382151-f32e-4a5a-9b74-1de057499237)


```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/b193f7fe-d80b-45b7-a9f4-1bea3fcec899)


```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/0d4792c8-f1ea-44c6-bce6-70f725363a75)
```py
tips.time.unique()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/98f7d5ab-4712-4844-812c-4715c65b96a2)
```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/8ca322ba-d9c9-4b94-b50e-876db6259e4f)

```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/PriyankaAnnadurai/EXNO-4-DS/assets/118351569/74662e12-9184-43e6-8754-32337b8aa010)


## RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
