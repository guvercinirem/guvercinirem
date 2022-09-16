#PROBLEM

#MAAŞ BİLGİLERİ VE 1986 YILINA AİT KARİYER İSTATİSTİKLERİ PAYLAŞILAN
#BEYZBOL OYUNCULARININ MAAŞ TAHMİNLERİ İÇİN BİR MAKİNE ÖĞRENMESİ MODELİ GELİŞTİRİN


#GÖREV1: VERİ ÖN İŞLEME
#GÖREV2: ÖZELLİK MÜHENDİSLİĞİ  işlemlerini gerçekleştirerek
#maaş tahmin modeli geliştiriniz.

#Kütüphaneler



import numpy as np
import pandas as pd

import warnings
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,cross_val_score,validation_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsClassifier


from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore",category=FutureWarning)
warnings.simplefilter(action="ignore",category=ConvergenceWarning)
warnings.simplefilter(action="ignore",category=SettingWithCopyWarning)

#GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTİONAL EDA)

# 1.Exploratory Data Analysis

# 1.GENEL RESİM
# 2.KATEGORİK DEĞİŞKEN ANALİZİ
# 3.SAYISAL DEĞİŞKEN ANALİZİ
# 4.KOLERASYON ANALİZİ


###===###===###===###===
#1. GENEL RESİM
###===###===###===###===

df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/hitters.csv")

def check_df(dataframe,head=5): #genel resmi görmek amacıyla
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

check_df(df)


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

cat_cols, num_cols, cat_but_car = grab_col_names(df)


###===###===###===###===
#2.KATEGORİK DEĞİŞKEN ANALİZİ (ANALYSIS OF CATEGORICAL VARIABLES)
###===###===###===###===

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("=====================================")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df,col,plot=True)


###===###===###===###===
#3.SAYISAL DEĞİŞKEN ANALİZİ (ANALYSIS OF NUMERICAL VARIABLES)
###===###===###===###===


def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df,col,plot=True)

###===###===###===###===
#4.HEDEF DEĞİŞKEN ANALİZİ (ANALYSIS OF TARGET VARIABLE)
###===###===###===###===




def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}),end="\n\n\n")
for col in cat_cols:
    target_summary_with_cat(df,"Salary",col)


###===###===###===###===
#4.KOLERASYON ANALİZİ (ANALYSIS OF CORRELATION)
###===###===###===###===


def high_correlated_cols(dataframe,plot=False,corr_th=0.90):
    corr=dataframe.corr()
    corr_matrix=corr.abs()
    upper_triangle_matrix=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_))
    drop_list=[col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize":(15,15)})
        sns.heatmap(corr,cmap="RdBu")
        plt.show()
    return drop_list
high_correlated_cols(df,plot=True)


#GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTİONAL EDA)

#1. OUTLIERS
#2.MISSING VALUES
#3.FEATURE EXTRACTION(özellik çıkarma)
#4.ENCODING LABEL-ONEHOT-RARE
#5.FEATURE SCALING (özellik ölçeklendirme)


########################################
#1. OUTLIERS(AYKIRI DEĞER AYIKLAMA)
#######################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#outlierların hepsi false döndü, tüm aykırı değerleri baskıladık.




########################################
#1. MISSING VALUES
#######################################


def missing_values_table(dataframe,na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss=dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df=pd.concat([n_miss,np.round(ratio,2)],axis=1,keys=["n_miss","ratio"])
    print(missing_df,end="\n")
    if na_name:
        return na_columns
missing_values_table(df)
missing_values_table(df,True)
#BAĞIMLI DEĞİŞKENDEKİ NA leri doldururken çok dikkatli olmak gerekir

df.dropna(inplace=True)



########################################
#1. FEATURE EXTRACTION(ÖZELLİK ÇIKARIMI)
#######################################


new_num_cols=[col for col in num_cols if col!="Salary"]

df[new_num_cols]=df[new_num_cols] + 0.0000000001

df["NEW_Hits"]=df["Hits"]/df["CHits"]

df["NEW_RBI"]=df["RBI"]/df["CRBI"]

df["NEW_WALKS"]=df["Walks"]/df["CWalks"]

#Daha bir sürü değişken türetilebilir.


########################################
#. ONE-HOT-ENCODING
#######################################


def one_hot_encoder(dataframe,categorical_cols,drop_first=True):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,cat_cols,drop_first=True)



df.head()



########################################
#. FEATURE SCALING (ÖZELLİK ÖLÇEKLENDİRME)
#######################################

cat_cols,num_cols,cat_but_car=grab_col_names(df)

num_cols=[col for col in num_cols if col not in "Salary"]
scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()


########################################
#. BASE MODEL
#######################################


#BASE MODEL KURULUMU
y=df["Salary"]
X=df.drop(["Salary"],axis=1)

models=[("LR",LinearRegression()),
         ("Ridge",Ridge()),
        ("Lasso",Lasso()),
         ("ElasticNet",ElasticNet()),
         ("KNN",KNeighborsRegressor()),
         ("CART",DecisionTreeRegressor()),
         ("RF",RandomForestClassifier()),
         ("SVR",SVR()),
         ("GBM",GradientBoostingRegressor()),
         ("XGBoost",XGBRegressor(objective="reg:squarederror")),
         ("LIGHTGBM",LGBMRegressor()),
         ("CatBoost",CatBoostRegressor(verbose=False))]


for name,regressor in models:
    rmse=np.mean(np.sqrt(-cross_val_score(regressor,X,y,cv=10,scoring="neg_mean_squared_error")))
    print(f"RMSE:{round(rmse,4)} ({name})")

#RANDOM FORESTS

rf_model=RandomForestRegressor(random_state=17)

rf_params={"max_depth":[5,8,15,None],
           "max_features":[5,7,"auto"],
           "min_samples_split":[8,15,20],
           "n_estimators":[200,500]}



rf_best_grid=GridSearchCV(rf_model,rf_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
rf_final=rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(X,y)
rmse=np.mean(np.sqrt(-cross_val_score(rf_final,X,y,cv=10,scoring="neg_mean_squared_error")))
rmse

#GBM MODEL

gbm_model=GradientBoostingRegressor(random_state=17)
gbm_params={"max_depth":[3,8],
           "learning_rate":[0.01,0.1],
           "subsample":[1,0.5,0.7],
           "n_estimators":[500,1000]}
gbm_best_grid=GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
gbm_final=gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17).fit(X,y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final,X,y,cv=10,scoring="neg_mean_squared_error")))

#LightGBM

lgbm_model=LGBMRegressor(random_state=17)
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

lgbm_best_grid=GridSearchCV(lgbm_model,lightgbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
lgbm_final=lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17).fit(X,y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final,X,y,cv=10,scoring="neg_mean_squared_error")))


catboost_model = CatBoostRegressor(random_state=17, verbose=False)
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


##########CatBoost##############



catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(catboost_final,X,y,cv=10,scoring="neg_mean_squared_error")))


# Automated Hyperparameter Optimization

rf_params={"max_depth":[5,8,15,None],
           "max_features":[5,7,"auto"],
           "min_samples_split":[8,15,20],
           "n_estimators":[200,500]}
gbm_params={"max_depth":[3,8],
           "learning_rate":[0.01,0.1],
           "subsample":[1,0.5,0.7],
           "n_estimators":[500,1000]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


regressors = [("RF", RandomForestClassifier(),rf_params),
              ("GBM",GradientBoostingRegressor(),gbm_params),
               ("LightGBM", LGBMRegressor(),lightgbm_params),
               ("CatBoost", CatBoostRegressor(),catboost_params)]

best_models={}
for name,regressor,params in regressors:
    print(f"###############{name}###############")
    rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))

    print(f"RMSE:{round(rase,4)}({name}")
    gs_best=GridSearchCV(regressor,params,cv=3,n_jobs=-1,verbose=False).fit(X,y)

    final_model=regressor.set_params(**gs_best.best_params_)
    rase=np.mean(np.sqrt(-cross_val_score(final_model,X,y,cv=10,scoring="neg_mean_squared_error")))
    print(f"RMSE(After) :{round(rase, 4)}({name}")
    print(f"{name} best params : {gs_best.best_params}",end="\n\n")
    best_models[name]=final_model


#FETAURE IMPORTANCES




def plot_importance(model,features,num=len(X),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,"Feature":features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
plot_importance(rf_final,X)
plot_importance(gbm_final,X)
plot_importance(lgbm_final,X)
plot_importance(catboost_final,X)


##############################
##ANALYZING MODEL COMPLEXITY WITH LEARNING CURVES

def val_curve_params(model,X,y,param_name,param_range,scoring="roc_auc",cv=10):
    train_score,test_score=validation_curve(model,X=X,y=y,param_name=param_name,param_range=param_range,scoring=scoring,cv=cv)
    mean_train_score=np.mean(train_score,axis=1)
    mean_test_score=np.mean(test_score,axis=1)

    plt.plot(param_range,mean_train_score,label="Training Score",colr="b")
    plt.plot(param_range,mean_test_score,label="Validation Score",color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name} ")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

rf_val_params=[["max_depth",[5,8,15,20,30,None]],
          ["max_feature",[3,5,7,"auto"]],
          ["min_samples_split",[2,5,8,15,20]],
          ["n_estimators",[10,50,100,200,500]]]
for i in range(len(rf_val_params)):
    val_curve_params(rf_model,X,y,rf_val_params[i][0],rf_val_params[i][1],scoring="neg_mean_squared_error")


#rf_val_params[0][1]
#hitters ile maaş tahmin bitti










