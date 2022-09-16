
################################################
# KNN:
# KNN, Denetimli Öğrenmede sınıflandırma ve regresyon
# için kullanılan algoritmalardan biridir.
# En basit makine öğrenmesi algoritması olarak kabul edilir.
# Diğer Denetimli Öğrenme algoritmalarının aksine,
# eğitim aşamasına sahip değildir. Eğitim ve test hemen hemen aynı şeydir.
# Tembel bir öğrenme türüdür. Bu nedenle,
# kNN, geniş veri setini işlemek için gereken algoritma olarak ideal bir aday değildir.
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model


import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import GridSearchCV,cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################
df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/diabetes.csv")

df.shape
df.info()
df.describe().T

df["Outcome"].value_counts()
"""Out[26]: 0    500/ 1    268"""

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

X_scaled=StandardScaler().fit_transform(X)
X=pd.DataFrame(X_scaled,columns=X.columns)


################################################
# 3. Modeling & Prediction
################################################


knn_model=KNeighborsClassifier().fit(X,y)

random_user=X.sample(1,random_state=45)
knn_model.predict(random_user) #Out[37]: array([1], dtype=int64)



################################################
# 4. Model Evaluation
################################################


# Confusion matrix için y_pred:

y_pred=knn_model.predict(X)


# AUC için y_prob:

y_prob=knn_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))
# acc 0.83
# f1 0.74
roc_auc_score(y,y_prob)
## 0.90


cv_results=cross_validate(knn_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean() #0.73
cv_results["test_f1"].mean() #0.59
cv_results["test_roc_auc"].mean() #0.78


# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()
"""{'algorithm': 'auto',
 'leaf_size': 30,
 'metric': 'minkowski',
 'metric_params': None,
 'n_jobs': None,
 'n_neighbors': 5,
 'p': 2,
 'weights': 'uniform'}"""



################################################
# 5. Hyperparameter Optimization
################################################

knn_model=KNeighborsClassifier()

knn_params={"n_neighbors":range(2,50)}
knn_gs_best=GridSearchCV(knn_model,knn_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)

#Fitting 5 folds for each of 48 candidates, totalling 240 fits

knn_gs_best.best_params_ #Out[54]: {'n_neighbors': 17}

knn_final=knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_results=cross_validate(knn_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean() #0.76
cv_results["test_f1"].mean() #0.61
cv_results["test_roc_auc"].mean() #0.81


random_user=X.sample(1)
knn_final.predict(random_user)