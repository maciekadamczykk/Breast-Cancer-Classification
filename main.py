import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("breast_cancer.csv") 

print(df["Class"].nunique())

X = df.iloc[:,1:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=777)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled,y_train)

y_pred = lr.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
c_matrix = confusion_matrix(y_pred,y_test)
sns.heatmap(c_matrix,annot=True)
plt.show()
acc = accuracy_score(y_test,y_pred)
print(acc)