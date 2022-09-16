######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.



# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


######################################################
# Exploratory Data Analysis
######################################################


df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/diabetes.csv")

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

    print()

check_df(df)


##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()
sns.countplot(x="Outcome",data=df)
plt.show()

##########################
# Feature'ların Analizi
##########################
df.head()
df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()
df.head()
df["Pregnancies"].hist(bins=20)
plt.xlabel("Pregnancies")
plt.show()
df["Pregnancies"].max()


#Numeriklerin görselleştirilmesi

def plot_numerical_col(dataframe,numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

#hedef değişkeni çıkartmak istersem:

cols=[col for col in df.columns if "Outcome" not in col]


for col in cols:
    plot_numerical_col(df,col)


df.describe().T



##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies":"mean"})

"""         Pregnancies
Outcome             
0           3.298000
1           4.865672  """

def target_summary_with_num(dataframe,target,numerical_col):
  print(dataframe.groupby(target).agg({numerical_col:"mean"}, end="\n\n\n"))
for col in cols:
    target_summary_with_num(df,"Outcome",col)



######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################

df.shape
df.info()
df.isnull().sum() #NA değer yok gibi görünüyor
df.describe().T #ama glucose veya BMI si 0 olan değerler var.
#bu veri setinde hamilelik dışında 0 değer olamaz. demekki eksik değerler 0 olarak girilmiş.

for col in cols:
    print(col,check_outlier(df,col))

#Insulin True

replace_with_thresholds(df,"Insulin")

#Eksik değer kalmadığına göre standartlaştırma işlemine geçebilirim.

#Şimdi hedef değişken hariç tüm numerik değişkenleri standartlaştıracağım.

for col in cols:
    df[col]=RobustScaler().fit_transform(df[[col]])


df.head()



######################################################
# Model & Prediction
######################################################


y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

log_model=LogisticRegression().fit(X,y)

log_model.intercept_[0] #Out[37]: -1.2343958783485596
log_model.coef_[0][0]#Out[38]: 0.5990678484490783


y_pred=log_model.predict(X)
y_pred[0:10]
y[0:10]










