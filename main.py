import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("breast_cancer.csv") 

X = df.iloc[:,1:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()