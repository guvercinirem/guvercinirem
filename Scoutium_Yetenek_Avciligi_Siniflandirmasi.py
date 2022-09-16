#YAPAY ÖĞRENME İLE YETENEK AVCILIĞI SINIFLANDIRMASI

#İş problemi : Scoutlar tarafındanizlenen futbolcuların özelliklerine verilen puanlara göre,
#oyuncuların hangi sınıf(average,highlighted)oyuncu olduğunu tahminleme

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
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

pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)



#Görev1: scotium_attributes.csv ve scotium_potential_labels.csv dosyalarını okutunuz


df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/scoutium_attributes.csv",sep=";")
df.head()
df.shape
df2=pd.read_csv("MIUUL/MACHİNE_LEARNING_/scoutium_potential_labels.csv",sep=";") #veri seti noktalı virgülle ayrıldığı için böyle yaptık
#Görev2: okytmuş olduğunuz dosyaları merge fonksiyonunu kullanarak birleştiriniz. ("task_response_id,"evaluator_id","player_id","match_id" 4 adet değişken üzerinden birleştirme işlemi yapınız.
df2.head()
df.shape
dff=pd.merge(df,df2,how="left",on=['task_response_id','match_id','evaluator_id','player_id'])
dff.head()
dff.shape




#Görev3: position_id içerisindeki Kaleci(1)sınıfını veri setinden kaldırınız.

dff=dff[dff["position_id"]!=1]

#Görev4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
#(below_average sınıfı veri setinin yüzde 1 ini oluşturuyor)
dff=dff[dff["potential_label"]!="below_average"]

#Görev4:Oluşturduğunuz veri setinden "pivot_table" fonksşşyonu kullanarak bir tablo oluşturun.
#Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

    #Adım1:
    #İndekslerde "player_id","position_id", ve "potential_label",sütunlarda "attribute_id",değerlerde
    #scoutların oyuncularına verdiği puan "attribute_value olacak şekilde pivot table oluşturunuz.

pt=pd.pivot_table(dff,values="attribute_value",columns="attribute_id",index=["player_id","position_id","potential_label"])

#Adım2: "resert_index fonksiyonunu kullanarak, index hatasından kurtulunuz.
# ve "attribute_id" sütunlarının isimlerini stringe çeviriniz.

pt=pt.reset_index(drop=False)
pt.head()
pt.columns=pt.columns.map(str)


#Görev6: Label encoder fonksiyonunu kullanarak ,
# potential_label kategorilerini (average,highlighted) sayısal olarak ifade ediniz.


le=LabelEncoder()
pt["potential_label"]=le.fit_transform(pt["potential_label"])

#Görev7: Sayısal Değişken kolonlarınnı "num_cols" adıyla bir listeye atınız.

num_cols=pt.columns[3:]

#grab_col_names i de çağırabilirdim ama veri setine hakim olduğum için böyle yaptım.

#Görev8: Kaydettiğiniz bütün num_cols ları değişkenlerdeki veriyi ölçeklendirmek için standart scale uygulayınız.

scaler=StandardScaler()
pt[num_cols]=scaler.fit_transform(pt[num_cols])

#Görev9: Elimizdeki veri seti üzerinden minimum hata ile
# furbolcuların potensiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.

y=pt["potential_label"]
X=pt.drop(["potential_label","player_id"],axis=1)

models=[("LR",LogisticRegression()),
       ("KNN", KNeighborsClassifier()),
       ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ("SVC",SVC()),
        ("Adaboost",AdaBoostClassifier()),
        ("LightGBM",LGBMClassifier()),
        ("CatBoost",CatBoostClassifier())]

for name,model in models:
    print(name)
    for score in ["roc_auc","f1","precision","recall","accuracy"]:
        cvs=cross_val_score(model,X,y,scoring=score,cv=10).mean()
        print(score+"score:"+str(cvs))


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
model=LGBMClassifier()
model.fit(X,y)
plot_importance(model,X)
