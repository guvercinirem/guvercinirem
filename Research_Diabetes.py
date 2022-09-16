################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function


################################################
# 1. Exploratory Data Analysis
################################################

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)




def check_df(dataframe,head=5):
    print("*******shape********")
    print(dataframe.shape)
    print("**********types**********")
    print(dataframe.dtypes)
    print("**********head**********")
    print(dataframe.head(head))
    print("**********tail**********")
    print(dataframe.tail(head))
    print("**********NA**********")
    print(dataframe.isnull().sum())
    print("**********Quantiles**********")
    print(dataframe.describe([0,0.05,0.95,0.99,1]).T)

    print()


def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("=====================================")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}),end="\n\n\n")

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")


def correlation_matrix(df,cols):
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig=sns.heatmap(df[cols].corr(),annot=True,linewidths=0.5,annot_kws={"size":12},linecolor="w",cmap="RdBu")
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/diabetes.csv")

check_df(df)

#Değişkenlerin türlerinin ayrıştırılması

cat_cols,num_cols,cat_but_car=grab_col_names(df,cat_th=5,car_th=20)

#bu veri setinde 1 tane kategorik değişken var ama daha fazla olsaydı

for col in cat_cols:
    cat_summary(df,col)


#Sayısal değişkenlerin incelenmesi

df[num_cols].describe().T

#Sayısal değişkenlerin grafiğini oluşturmak istersem :


#for col in num_cols:
    #num_summary(df,col,plot=True)

#Sayısal değişkenlerin birbiri ile kolerasyonu :

correlation_matrix(df,num_cols)



#Target ile sayısal değişkenlerin incelenmesi

for col in num_cols:
    target_summary_with_num(df,"Outcome",col)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

#AYKIRI DEĞERLER İÇİN KULLANDIĞIMIZ 3 FONKSİYON

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name,q1=0.25,q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1,q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#bu veri setinde hedef değişkenimiz dışında kategorik değişken olmamasına rağmen
#feature engineering kapsamında yeni değişkenler türeteceğiz. onların içinde kategorikler olacak.
#o yüzden one_hot encoder fonksiyonunu çağırıyoruz.
#one-hot encoderı aynı zamanda label encoder olarak da kullanabilmek için
#drop_first argümanını TRUE yaptık.



def one_hot_encoder(dataframe,categorical_cols,drop_first=True):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,cat_cols,drop_first=True)

#DEĞİŞKEN İSİMLERİNİ BÜYÜTMEK

df.columns=[col.upper() for col in df.columns]
df.head()

#Şimdi yeni değişkenler türetmek istiyorum

#glikoz değişkeni direk diabet ile ilgili olduğu için
#139 a kadar normal, üstüne prediabet isimlendirmesi yapacağım


df["NEW_GLUCOSE_CAT"]=pd.cut(x=df["GLUCOSE"],bins=[-1,139,200],labels=["normal","prediabetes"])

#Şimdilik Glükoz değişkenini çıkartmıyoruz.Sonra ağaç yöntemini oluştururken karar noktamız olacak.

#Şimdi yaş değişkeni üzerinden değişkenler oluşturalım.

df.loc[(df["AGE"]<35),"NEW_AGE_CAT"]="young"
df.loc[(df["AGE"]>=35)&(df["AGE"]<=55),"NEW_AGE_CAT"]="middleage"
df.loc[(df["AGE"]>55),"NEW_AGE_CAT"]="old"


#BMI üzerinden yeni değişkenler türetelim.

df["NEW_BMI_RANGE"]=pd.cut(x=df["BMI"],bins=[-1,18.5,24.9,29.9,100],labels=["underweight","healty","overweight","obese"])

#BloodPressure üzerinden yeni değişkenler türetelim.


df["NEW_BLOODPRESSURE"]=pd.cut(x=df["BLOODPRESSURE"],bins=[-1,79,89,123],labels=["normal","hs1","hs3"])

#yeni değişkenleri oluşturduktan sonra tekrar bir kontrol gerekir.

check_df(df)
cat_cols,num_cols,cat_but_car=grab_col_names(df,cat_th=5,car_th=20)
for col in cat_cols:
    cat_summary(df,col)

for col in cat_cols:
    target_summary_with_cat(df,"OUTCOME_1",col)

#Outcome hedef değişkenin de analizini yapmış doğal olarak.

cat_cols=[col for col in cat_cols if "OUTCOME_1" not in col]


#Makine öğrenmesi yöntemleri bizden standart bir format bekliyor.
#standartlaştırmayı da:
# label encode ederek, 1,2,3 diyerek sınıflandırabilir.
#veya one-hot encode yöntemi ile indexi değişkene atayabiliriz.

df=one_hot_encoder(df,cat_cols,drop_first=True)
check_df(df)

#yeni encodelardan sonra tekrar büyütme yapmamız lazım.

df.columns=[col.upper() for col in df.columns]

df.head()

#SON GÜNCEL DEĞİŞKEN TÜRLERİNİ TUTUYORUM.

cat_cols,num_cols,cat_but_car=grab_col_names(df,cat_th=5,car_th=20)

cat_cols=[col for col in cat_cols if "OUTCOME_1" not in col]


for col in num_cols:
    print(col,check_outlier(df,col,0.05,0.95))

""" niye 0.05,0.95 aldım, çünkü çok değişkenli etki aykırı olmayan bir durumu aykırı hale getirebilir.
boxplot yöntemi bunu tıraşlayabilir. 17 yaş aykırı değildir, 3 hamilelik aykırı değildir ama 17 yaşında 3 hamilelik aykırıdır."""

""" Insulin True geldi."""


replace_with_thresholds(df,"INSULIN")

"""STANDARTLAŞTIRMA"""

X_scaled=StandardScaler().fit_transform(df[num_cols])
df[num_cols]=pd.DataFrame(X_scaled,columns=df[num_cols].columns)

y=df["OUTCOME_1"]
X=df.drop(["OUTCOME_1"],axis=1)

check_df(df)


"""yaptığımız işlemleri fonksiyonlaştıralım"""

def diabetes_data_prep(dataframe):
    dataframe.columns=[col.upper() for col in dataframe.columns]

    #Glucose

    dataframe["NEW_GLUCOSE_CAT"] = pd.cut(x=df["GLUCOSE"], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    #AGE

    dataframe.loc[(df["AGE"] < 35), "NEW_AGE_CAT"] = "young"
    dataframe.loc[(df["AGE"] >= 35) & (df["AGE"] <= 55), "NEW_AGE_CAT"] = "middleage"
    dataframe.loc[(df["AGE"] > 55), "NEW_AGE_CAT"] = "old"

    # BMI

    dataframe["NEW_BMI_RANGE"] = pd.cut(x=df["BMI"], bins=[-1, 18.5, 24.9, 29.9, 100],
                                 labels=["underweight", "healty", "overweight", "obese"])

    # BloodPressure

    dataframe["NEW_BLOODPRESSURE"] = pd.cut(x=df["BLOODPRESSURE"], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs3"])

    cat_cols,num_cols,cat_but_car=grab_col_names(dataframe,cat_th=5,car_th=20)
    cat_cols=[col for col in cat_cols if "OUTCOME_1" not in col]
    dataframe=one_hot_encoder(dataframe,cat_cols,drop_first=True)
    cat_cols,num_cols,cat_but_car=grab_col_names(dataframe,cat_th=5,car_th=20)
    cat_cols=[col for col in cat_cols if "OUTCOME_1" not in col]

    replace_with_thresholds(dataframe,"INSULIN")

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    y = dataframe["OUTCOME_1"]
    X = dataframe.drop(["OUTCOME_1"], axis=1)
    return X,y


df=pd.read_csv("MIUUL/FEATURE_ENGINEERING_DATA_PREPROCESSING/diabetes.csv")

check_df(df)

X,y=diabetes_data_prep(df)

check_df(X)


# 3. Base Models


def base_models(X,y,scoring="roc_auc"):
    print("Base Models...")

    classifiers=[("LR",LogisticRegression()),
                 ("KNN",KNeighborsClassifier()),
                 ("SVC",SVC()),
                 ("CART",DecisionTreeClassifier()),
                 ("RF",RandomForestClassifier()),
                 ("Adaboost",AdaBoostClassifier()),
                 ("GBM",GradientBoostingClassifier()),
                 ("XGBoost",XGBClassifier(use_label_encoder=False,eval_metric="logloss")),
                 ("LightGBM",LGBMClassifier()),
                 #("CatBoost",CatBoostClassifier(verbose=False))
                 ]

    for name,classifier in classifiers:
        cv_results=cross_validate(classifier,X,y,cv=3,scoring=scoring)
        print(f'{scoring}:{round(cv_results["test_score"].mean(), 4)}({name})')


base_models(X,y)

#Hepsini hızlı bir şekilde sordum ve Logistik Regresyon daha başarılı gibi geldi, ama bilmiyorum.
#roc_auc:0.8409(LR)

#Scoring'i f1 ile değiştirsem bakalım ne olacak

base_models(X,y,scoring="f1")

#f1:0.6386(RF) en yüksek RF geldi bu sefer

base_models(X,y,scoring="accuracy")

#Başarılı bir şekilde modellerin ilk ham halleriyle sorgulama yaptık.


# 4. Automated Hyperparameter Optimization

knn_params={"n_neighbors":range(2,50)}

cart_params={"max_depth":range(1,20),
             "min_samples_split":range(2,30)}

rf_params={"max_depth":[8,15,None],
           "max_features":[5,7,"auto"],
           "min_samples_split":[15,20],
           "n_estimators":[200,300]}

xgboost_params={"learning_rate":[0.1,0.01],
           "max_depth":[5,8],
            "n_estimators":[100,200]}


lightgbm_params={"learning_rate":[0.01,0.1],
                 "n_estimators":[300,500]}

classifiers = [("KNN", KNeighborsClassifier(),knn_params),
               ("CART", DecisionTreeClassifier(),cart_params),
               ("RF", RandomForestClassifier(),rf_params),
               ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss"),xgboost_params),
               ("LightGBM", LGBMClassifier(),lightgbm_params)]

def hyperparameter_optimization(X,y,cv=3,scoring="roc_auc"):
    print("Hyperparameter Optimization....")

    best_models={}
    for name,classifier,params in classifiers:
        print(f"#############{name} #############")

        cv_results=cross_validate(classifier,X,y,cv=cv,scoring=scoring)

        print(f"{scoring}(Before):{round(cv_results['test_score'].mean(),4)}")

        gs_best=GridSearchCV(classifier,params,cv=cv,n_jobs=-1,verbose=False).fit(X,y)
        final_model=classifier.set_params(**gs_best.best_params_)
        cv_results=cross_validate(final_model,X,y,cv=cv,scoring=scoring)
        print(f"{scoring}(After): {round(cv_results['test_score'].mean(),4)}")
        print(f"{name} best params: {gs_best.best_params_}",end="\n\n")
        best_models[name]=final_model
    return best_models


best_models=hyperparameter_optimization(X,y)

#En iyisi şuan için RF gibi görünüyor ama biz çok toy aralıklar verdik.



# 5. Stacking & Ensemble Learning

#Bir arada bir çok modeli kullanmak.
#Tahmin performansını iyileştirmek_hepsinin gücünü bir araya getirerek


def voting_classifier(best_models,X,y):
    print("Voting Classifier...")

    voting_clf=VotingClassifier(estimators=[("KNN",best_models["KNN"]),("RF",best_models["RF"]),
                                           ("LightGBM",best_models["LightGBM"])],
                               voting="soft").fit(X,y)
    cv_results=cross_validate(voting_clf,X,y,cv=3,scoring=["accuracy","f1","roc_auc"])
    print(f"Accuracy:{cv_results['test_accuracy'].mean()}")
    print(f"F1Score:{cv_results['test_f1'].mean()}")
    print(f"ROC_AUC:{cv_results['test_roc_auc'].mean()}")
    return voting_clf


voting_clf=voting_classifier(best_models,X,y)

"""Voting Classifier...
Accuracy:0.7734375
F1Score:0.6397058823529412
ROC_AUC:0.8363008306447135
"""

# 6. Prediction for a New Observation


"""veri setinden rastgele bir gözlem biri seçip, tahmin işlemini yapacağız"""

X.columns
random_user=X.sample(1,random_state=45)
voting_clf.predict(random_user)

"""bu modeli kaydetmek istiyorsam da joblib ten faydalanacağım"""


joblib.dump(voting_clf,"voting_clf2.pkl")

#çalışma dizinini yandaki project bölümü yenilediğimizde
# voting_clf2pkl gelmiş olmalı


new_model=joblib.load("voting_clf2.pkl")
new_model.predict(random_user)

#Out[36]: array([1], dtype=uint8)



## 7. Pipeline Main Function


def main():
    df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/diabetes.csv")
    X,y=diabetes_data_prep(df)
    base_models(X,y)
    best_models=hyperparameter_optimization(X,y)
    voting_clf=voting_classifier(best_models,X,y)
    joblib.dump(voting_clf,"voting_clf.pkl")
    return voting_clf

if __name__=="__main__":
    main()




















