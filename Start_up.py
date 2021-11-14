import pandas as pd
import csv
from sklearn import preprocessing
df = pd.read_csv("investments_VC.csv",encoding= 'unicode_escape')

del df['permalink']
del df['homepage_url']
del df['category_list']

df.columns

print(df.columns)

df_algo = df[[' market ',' funding_total_usd ','status','country_code','funding_rounds','seed','venture','equity_crowdfunding','angel','grant']]

df_algo.dropna()

one_hot = pd.get_dummies(df_algo[' market '])

df_algo = df_algo.join(one_hot)

df_algo['funding'] = df_algo[' funding_total_usd '].str.strip()

df_algo['funding'] = df_algo['funding'].str.replace(',', '')

del df_algo[' funding_total_usd ']

df_algo = df_algo.dropna()

df_algo =df_algo[~df_algo['funding'].str.startswith('-')]

#del df_algo[' market ']
del df_algo['country_code']

del df_algo[' market ']

from sklearn.model_selection import train_test_split

#Dropping target column
X = df_algo
y = df_algo['status']


le = preprocessing.LabelEncoder()
le.fit(df['status'])
df['status']=le.transform(df['status'])
df=df['status'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

# Apply Logistic Reg model

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(C=100,fit_intercept=False)


logmodel = LogisticRegression(C=100,fit_intercept=False)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

predictions

#Random Forest

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

RandomForestClassifier(random_state=42)

predictions = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
