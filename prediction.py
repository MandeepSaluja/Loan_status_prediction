import pandas as pd

df = pd.read_csv("loan_status.csv")
df.head()

df.shape # (614, 13)

df.isnull().sum()
# Many columns are having null values in it. Will handle them later. Let’s check other characteristics of this dataset.

df.dtypes
df.describe()

clean_df = df.dropna()
clean_df.shape #(480, 13) We didn’t lose a lot of rows. 480 is still fine.

clean_df.isnull().sum()

'''We are just going to simple replace the categorical values with 0s and 1s using the replace() method.

For example, If the person is “Male” we can replace it with 1 and if “Female” we can replace it with 0. 
Similarly, If the person is a “Graduate” we can replace it with 1 and if the person is “Not Graduate” we can replace it with 0 and so on.'''

clean_df.replace({"Gender":{"Male":1, "Female":0}}, inplace=True)
clean_df.replace({"Married":{"Yes":1, "No":0}}, inplace=True)
clean_df.replace({"Education":{"Graduate":1, "Not Graduate":0}}, inplace=True)
clean_df.replace({"Self_Employed":{"Yes":1, "No":0}}, inplace=True)
clean_df.replace({"Property_Area":{"Rural":0, "Semiurban":1, "Urban":2}}, inplace=True)
clean_df.replace({"Loan_Status":{"Y":1, "N":0}}, inplace=True)


# We will drop the drop the “Loan_ID” column.
clean_df = clean_df.drop(columns='Loan_ID')

#See the “Dependents” column it is still in object data type.

clean_df.Dependents.value_counts()

"""
OUTPUT:

Dependents
0     274
2      85
1      80
3+     41
"""

'''Okay, it seems like there is a value “3+” in the “Dependents” column which is making this column an object datatype.
Let’s replace that value with 3 and then change the datatype using astype() method..'''

clean_df.replace(to_replace='3+', value=3, inplace=True)
clean_df['Dependents'] = clean_df['Dependents'].astype(int)

# EDA

import plotly.express as px

px.imshow(clean_df.corr())

import seaborn as sns

sns.countplot(data=clean_df, x = 'Gender', hue='Loan_Status')

'''We observed that the “Credit_History” column has nearly 50% correlation with the “Loan_Status” column.

It seems like men tends to get more loans than women.'''

y = clean_df['Loan_Status']
X = clean_df.drop(columns='Loan_Status')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=234)

'''when splitting a dataset into training and test sets, using stratify=y ensures that the proportion of different classes in the target variable y is preserved in both the training and test sets.
This is particularly useful in classification problems where the target variable might have an imbalanced distribution.'''

X_train.shape, y_train.shape #((384, 11), (384,))

X_test.shape, y_test.shape #((96, 11), (96,))

#Support Vector Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_prediction)
svc_accuracy # 0.78125

confusion_matrix(y_test, svc_prediction)

"""
OUTPUT:

array([[10, 20],
       [ 1, 65]])
"""

#There is 10 TP, 65 TN, 20 FP, AND 1 FN. It’s a good stats.

#Below is the classification report including all the metrics like recall, precision, and f1-score.

print(classification_report(y_test, svc_prediction)) 

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)
dtc_accuracy = accuracy_score(y_test, dtc_prediction)
dtc_accuracy # 0.65625

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
rfc_accuracy = accuracy_score(y_test, rfc_prediction)
rfc_accuracy # 0.7916666666666666

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_prediction)
lr_accuracy # 0.7916666666666666


