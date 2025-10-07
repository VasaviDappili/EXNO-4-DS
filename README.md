# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME : DAPPILI VASAVI
REGISTER NUMBER : 212223040030
```
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"], engine='python')
data
```
<img width="1446" height="421" alt="image" src="https://github.com/user-attachments/assets/95240209-e427-4987-b20b-9be2424f874c" />

```
data.isnull().sum()
```
<img width="274" height="494" alt="image" src="https://github.com/user-attachments/assets/baf2b933-1bec-4bb7-98f4-dea60730c5be" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1417" height="425" alt="image" src="https://github.com/user-attachments/assets/a78c3092-60c7-4657-9209-d7ebb3ea8ca7" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1450" height="430" alt="image" src="https://github.com/user-attachments/assets/55c12ae8-9fec-4589-bb85-93827e7d24e3" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="1135" height="336" alt="image" src="https://github.com/user-attachments/assets/ebc8a10e-9a82-4faf-a547-aaa2d52adede" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="443" height="422" alt="image" src="https://github.com/user-attachments/assets/49e62cc7-2f4d-4232-985c-76a124332f50" />

```
data2
```
<img width="1333" height="421" alt="image" src="https://github.com/user-attachments/assets/d5e3c8e7-6c34-4dbc-8169-95a6b529c772" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1853" height="274" alt="image" src="https://github.com/user-attachments/assets/6a49185e-9f4c-4a53-9dae-6b8ee2d5ceb0" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1800" height="49" alt="image" src="https://github.com/user-attachments/assets/ee1be17e-d122-4dec-ba47-6a00448f1e77" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1781" height="49" alt="image" src="https://github.com/user-attachments/assets/18a1292e-0bae-44fe-8902-31995657a1ba" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="247" height="50" alt="image" src="https://github.com/user-attachments/assets/9ebc91e7-bb8a-443f-a814-2c02338d0b6f" />

```
x=new_data[features].values
print(x)
```
<img width="515" height="180" alt="image" src="https://github.com/user-attachments/assets/f15a4148-3a80-4591-a8b3-0e0b431e2b7f" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="369" height="95" alt="image" src="https://github.com/user-attachments/assets/d8ee5b39-46f1-4cb1-99f8-cd61cc0e75de" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="199" height="59" alt="image" src="https://github.com/user-attachments/assets/a8dc2e87-1722-4d95-a79b-ee3e3adaf00e" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="251" height="41" alt="image" src="https://github.com/user-attachments/assets/eecd24aa-a579-4309-88ea-47611a29a487" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="339" height="39" alt="image" src="https://github.com/user-attachments/assets/2c7bcd4e-6b27-4dc7-88f6-83a95b64a72d" />

```
data.shape
```
<img width="191" height="42" alt="image" src="https://github.com/user-attachments/assets/9516b1f1-0126-4a58-8718-5cfea4c02a13" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
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
<img width="1779" height="118" alt="image" src="https://github.com/user-attachments/assets/cc3bb235-3867-4cb1-9e74-3533fe86b4cc" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="628" height="254" alt="image" src="https://github.com/user-attachments/assets/85ae68c3-16f9-47a6-9ebf-bf6dfbdb4bd1" />

```
tips.time.unique()

```
<img width="510" height="64" alt="image" src="https://github.com/user-attachments/assets/2c0b6066-4c09-4469-a942-cc7930a23ecf" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="279" height="106" alt="image" src="https://github.com/user-attachments/assets/fc781599-ad76-4d80-b4b2-892b5df392c5" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="449" height="65" alt="image" src="https://github.com/user-attachments/assets/83489f92-1f5a-4c39-87c4-405dc8f7665f" />


# RESULT:
  Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
  save the data to a file is been executed.
