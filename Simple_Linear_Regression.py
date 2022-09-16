
#Sales Prediction with Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.float_format",lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/advertising.csv")

df.shape
df.info()

X=df[["TV"]]
y=df[["sales"]]


#MODEL

reg_model=LinearRegression().fit(X,y)

#y_hat=b+w*x

#sabit(b-bias)

reg_model.intercept_[0]

#ağırlık,x in katsayısı w1

reg_model.coef_[0][0]

#TAHMİN İŞLEMLERİ


#soru: 150 birimlik TV harcaması olsa ne kadar satış olması beklenir


reg_model.intercept_[0] + reg_model.coef_[0][0]*150


#soru: 500 birimlik TV harcaması olsa ne kadar satış olması beklenir

reg_model.intercept_[0] + reg_model.coef_[0][0]*500


df.describe().T #TV nin maksimumu 296


#MODELİN GÖRSELLEŞTİRİLMESİ

g=sns.regplot(x=X,y=y,scatter_kws={"color":"b","s":9},
              ci=False,color="r") #ci=False güven aralığı ekleme
g.set_title(f"Model Denklemi : Sales={round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0],2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()


#DOĞRUSAL REGRESYONDA TAHMİN BAŞARISI
y_pred=reg_model.predict(X)
mean_squared_error(y,y_pred)

#Out[2]: 10.512652915656757

#MSE 10 geldi, yorum yapamıyorum, karşılaştırmak adına ortalamalarına ve standart sapmalarına bakacağım
"""

y.mean()
Out[3]: 
sales   14.02
y.std()
Out[4]: 
sales   5.22

"""

#ortalamanın 14, std nin olduğu yerde mse 10 büyük gibi.


#RMSE

np.sqrt(mean_squared_error(y,y_pred))

#Out[5]: 3.2423221486546887

#MAE

mean_absolute_error(y,y_pred)

#Out[6]: 2.549806038927486

#R-KARE: doğrusal regresyon modelinde modelin başarısını gösteren çok önemli bir metriktir.

reg_model.score(X,y)
#Out[7]: 0.611875050850071

#bağımsız değişkenler bağımlı değişkenin %61 ini açıklayabilir.

#değişken sayısı arttıkça r kare şişmeye mahkumdur.
#düzeltilmiş r kare gerekli

#Basit doğrusal regresyon modeli yüksek tahmin başarısı içeren bir model olmamasına rağmen konunun temelidir.

#Ağaca dayalı modellerin başarı oranı daha iyidir.


#############################################3

#Multiple Linear Regression(çoklu doğrusal regresyon modeli)

df.head()
X=df.drop("sales",axis=1)

y=df[["sales"]]

##############MODEL#################

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

X_train.shape
y_train.shape
X_test.shape
X_test.shape

reg_model=LinearRegression()
reg_model.fit(X_train,y_train)

#sabit
reg_model.intercept_
#coefficients(w-weights)
reg_model.coef_

#TAHMİN

#Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir ?

#TV:30, radio:10, newspaper:40

"""reg_model.intercept_
Out[18]: array([2.90794702])
reg_model.coef_
Out[19]: array([[0.0468431 , 0.17854434, 0.00258619]])
"""

2.90794702 + 0.0468431*30 + 0.17*10 +0.002*40

#sales=2.90 + 0.04*TV + 0.17854434*radio +0.00258619*newspaper

#fonksiyonel hale getirmek için

yeni_veri=[[30],[10],[40]]
yeni_veri=pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

#Out[21]: array([[6.202131]])


#################################
#TAHMİN BAŞARISINI DEĞERLENDİRME

#TRAİN RMSE
#################################
y_pred=reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))

#Out[22]: 1.73690259014
#tek değişkende bu hata 3.24 idi bayağı düşmüş.


#TRAİN RKARE

reg_model.score(X_train,y_train)
# Out[23]: 0.8959372632325174  - bbu da yüzde 60 lardaydı


#yani yeni değişken eklendiğinde hata düşüyor.


#trainde baktık teste de bakalım.

#Test RMSE

y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


#Out[24]: 1.4113417558581587

#normalde beklenen test hatasının daha yüksek olması iken, test daha küçük çıkmış,beklenti dışı olmasına rağmen güzel bir durum.

#TEST RKARE

reg_model.score(X_test,y_test)


#train test diye ayırarak hold out yaptık.
#10 katlı CROSS validation da yapabilirdik.

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

#Out[26]: 1.6913531708051799

#5 katlısını da bir gözlemleyelim.
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

#Out[28]: 1.7175247278732086






































