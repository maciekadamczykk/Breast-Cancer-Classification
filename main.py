import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("breast_cancer.csv") 

X = df.iloc[:,1:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
c_matrix = confusion_matrix(y_pred,y_test)
sns.heatmap(c_matrix,annot=True)
plt.show()
acc = accuracy_score(y_pred,y_test)
print(acc)