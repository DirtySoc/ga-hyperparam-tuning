# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
rb=RobustScaler()

# %% Read in dataset.
data=pd.read_csv("impstroke.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)
data.head()

# %% Split data
X = data.drop("stroke",axis=1)
y = data["stroke"]
X.head()
# y.head()

# %% Data Feature Manipulation
from imblearn.over_sampling import RandomOverSampler
xfe, yfe = RandomOverSampler(sampling_strategy=0.25, random_state=11).fit_resample(X, y)
xref = xfe.copy(deep=True)
xtest = X.copy(deep=True)
xfe["Blood&Heart"]=xfe["hypertension"]*xfe["heart_disease"]
xtest["Blood&Heart"]=xtest["hypertension"]*xtest["heart_disease"]
xfe["Effort&Duration"] = xfe["work_type"]*(xfe["age"])
xtest["Effort&Duration"] = xtest["work_type"]*(xtest["age"])
xfe["Obesity"] = xfe["bmi"]*xfe["avg_glucose_level"]/1000
xtest["Obesity"] = xtest["bmi"]*xtest["avg_glucose_level"]/1000
xfe["AwfulCondition"] = xfe["Obesity"] * xfe["Blood&Heart"] * xfe["smoking_status"]
xtest["AwfulCondition"] = xtest["Obesity"] * xtest["Blood&Heart"] * xtest["smoking_status"]
# xfe["AwfulCondition"].unique()
xfe.head()

# %%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = [SVC(kernel='linear'),
          SVC(kernel='rbf'),
          SVC(kernel='poly',degree=2),
          RandomForestClassifier(n_jobs=-1,max_depth=10),
          RandomForestClassifier(n_jobs=-1,max_depth=30),
          KNeighborsClassifier(n_neighbors=4),
          KNeighborsClassifier(n_neighbors=8),
          LogisticRegression(),
          GaussianNB()]

names = ["SVM_Linear","SVM_RBF","SVM_Poly2","ShallowForest","DeepForest","4NN","8NN","LogReg","GaussianNB"]

xfesc = rb.fit_transform(xfe)
xtsc = rb.transform(xtest)

ffs_scores = pd.DataFrame(np.zeros((len(names),len(names))),columns=names,index=names)
ffsdata= dict()

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(models)):
    sel_name = names[i]
    ffs=SequentialFeatureSelector(direction='forward', n_jobs = -1, estimator=models[i])
    xffs = ffs.fit_transform(xfesc,yfe)
    xtfs = ffs.transform(xtsc)
    ffsdata[sel_name] = [xffs,xtfs]
    print(f"Finished Selection with {sel_name}\n")
    print(f"{ffs.n_features_to_select_} Features are Selected:\n{list(xfe.columns[ffs.support_])}\n")
    for j in range(len(models)):
        pred_name = names[j]
        model = models[j]
        model.fit(xffs,yfe)
        ypred = model.predict(xtfs)
        f1_score_val = f1_score(ypred,y)
        ffs_scores.loc[sel_name,pred_name] = f1_score_val
        print(f"F1_score with {pred_name}: {f1_score_val}")
        acc_score = accuracy_score(ypred,y)
        print(f"Accuracy with {pred_name}: {acc_score}")
        print(classification_report(ypred, y))
    print("\n\n\n")

# %%
