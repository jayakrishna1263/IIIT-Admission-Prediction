import pandas as pd
import numpy as np
import pickle

d=pd.read_csv("/Users//dasar//Desktop//project//admission.csv")
df=pd.DataFrame(d)

df['Govt/Private School']=df['Govt/Private School'].replace(['Govt school','Private'],[0.4,0])

df['Caste']=df['Caste'].replace(['OC','BC-A','BC-C','BC-D','BC-B','BC-E','SC','ST'],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

df['Gender']=df['Gender'].replace(['Male','Female'],[0.34,0.66])

df["Performance Score"] = df["Govt/Private School"] + df["Caste"] + df["Gender"] + df["10th GPA"]


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
from sklearn.model_selection import train_test_split

x=df.loc[:,["10th GPA","Govt/Private School","Caste","Gender"]]
y=df["Chance of Admit"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=45)

lr = LR.fit(x_train,y_train)

pickle.dump(lr, open('admi.pkl', 'wb'))
