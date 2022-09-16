##############
#KNN - K-NEAREST NEIGHBOURS (

#1.EXPLORATORY DATA ANALYSIS
#2.DATA PREPROCESSING & FEATURE ENGINEERING
#3.MODELING & PREDICTION
#4.MODEL EVALUATION
#5.HYPERPARAMETER OPTİMİZATION
#6.FINAL MODEL
import pandas as pd
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option("display.max_columns",None)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$
#1.EXPLORATORY DATA ANALYSIS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$

df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/diabetes.csv")
df.shape
df.describe().T
df["Outcome"].value_counts()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$
#2.DATA PREPROCESSING & FEATURE ENGINEERING
#$$$$$$$$$$$$$$$$$$$$$$$$$$$



#öncelikle elimizdeki bağımsız değişkenleri standartlaştıralım.
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_scaled=StandardScaler().fit_transform(X)
#x_scaled ı çalıştırdığımızda bir numpy array i şeklinde döndürüyor.
# o yüzden bir Dataframe e çevirip, sütun isimlerini ilk X verisinden çekebilirim.

X=pd.DataFrame(X_scaled,columns=X.columns)

#Şimdi standartlaştırılmış, ML formatına uygun değerler ile sütun isimleriyle birlikte X veri seti geldi.

#konumuz knn olduğu için bununla yetiniyoruz. dileyen eksik değer,aykırı değer,yeni değişken ekleme gibi gibi detaylandırabilir.



#$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 3. MODELING & PREDICTION
#$$$$$$$$$$$$$$$$$$$$$$$$$$$

knn_model=KNeighborsClassifier().fit(X,y)
random_user=X.sample(1,random_state=45) #veri setinden rastgele bir kullanıcı seçtim.
#bakalım bu kullanıcı diabet mi değil mi tahmin etsin KNN modeli

knn_model.predict(random_user)

# Out[8]: array([1], dtype=int64)  diyabetmiş



#$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 3. MODEL EVALUATION
#$$$$$$$$$$$$$$$$$$$$$$$$$$$

#bütün gözlem birimlerini tahmin etmek istersem

y_pred=knn_model.predict(X)

#AUC için y_prob:

y_prob=knn_model.predict_proba(X)[:,1]
print(classification_report(y,y_pred))

#AUC

roc_auc_score(y,y_prob)

#Out[11]: 0.9017686567164179 kayda değer bir başarı.

#acc:0.83
#f1:0.74
#AUC:0.90



#Modeli bütün veriyle test ettik. ama görmediği veriyle test etmemiz de gerekir.
#bunun için 20 e 80 hold out(sınama seti yaklaşımı-train-test- yöntemi veya k-katlı cross validation yöntemini deneyebiliriz.
#5 katlı cross validation ile devam edelim.


cv_results=cross_validate(knn_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean() #Out[15]: 0.733112638994992
cv_results["test_f1"].mean()#Out[16]: 0.5905780011534191
cv_results["test_roc_auc"].mean()#Out[17]: 0.7805279524807827


#veri setini cross validation ile çapraz doğrulama yaparak modellemizi uyguladığımızda
#tüm veri setini test etmek yerine daha farklı daha düşük sonuçlar aldık.
#çapraz doğrulamanın sonuçları daha doğrudur.5 kere model görmediği veriyle test edildiği için daha doğrudur.


#PEKİ BU BAŞARI SONUÇLARI NASIL ARTTIRABİLİR?

#1 VERİ BOYUTU YANİ GÖZLEM SAYISI ARTTIRILABİLİR.
#2.VERİ ÖN İŞLEME BÖLÜMÜ DETAYLANDIRILABİLİR.
#3.ÖZELLİK MÜHENDİSLİĞİ - YENİ DEĞİŞKENLER TÜRETİLEBİLİR.
#4.İLGİLİ ALGORİTMA İÇİN OPTİMİZASYONLAR YAPILABİLİR.
    #knn modelinin komşuluk sayısı hiperparametresinin ön tanımlı değeri değiştirilebilirdir.
    #değiştirip denedikçe, hiperparametre optimize edilir ve  final model kurulabilir.
knn_model.get_params()   #'n_neighbors': 5,    miş.






#$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 5. HYPERPARAMETER OPTIMIZATION
#$$$$$$$$$$$$$$$$$$$$$$$$$$$


knn_model=KNeighborsClassifier()
knn_model.get_params() #'n_neighbors': 5
#şimdi bir sözlük oluşturuyorum, get paramsın içindeki parametrenin ifade ediliiş tarzı aynı olacak şekilde.
knn_params={'n_neighbors': range(2,50)}

#peki en uygun parametreyi nasıl arayacağız gridSearchCV() ile.

GridSearchCV(knn_model,knn_params,cv=5,n_jobs=-1, verbose=1).fit(X,y) #j_jobs=-1 dendiğinde işlemciler maksimum performans ile arama yapar.
#verbouse argümanı yapılan aramalardan rapor bekliyor musun diye soruyor

#bunu da bir değişkene atayalım.

knn_gs_best=GridSearchCV(knn_model,knn_params,cv=5,n_jobs=-1, verbose=1).fit(X,y)



#output
#Fitting 5 folds for each of 48 candidates, totalling 240 fits
#yani : 48 tane denenecek olan hiperparametre değeri var aday olarak nitelendirilen.
#bu 48 tane adaya 5 katlı doğrulama yapılıyor.toplam 240 tane fit etme yani model kurma işlemi olacakmış.

knn_gs_best.best_params_
#Out[24]: {'n_neighbors': 17}

#yani ön tanımlı değer 5 yerine 17 komşuluk yaparsam model daha iyi sonuç verecek demektir.




#$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 6. FINAL MODEL
#$$$$$$$$$$$$$$$$$$$$$$$$$$$


knn_final=knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

#optime olmuş halde modeli kurduk.
#şimdi yine test hatasını yapmam gerek

cv_results=cross_validate(knn_final,
                          X,
                          y,
                          cv=5,
                          scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean() #Out[27]: 0.7669892199303965
cv_results["test_f1"].mean() #Out[28]: 0.6170909049720137
cv_results["test_roc_auc"].mean() #Out[29]: 0.8127938504542278


#bütün skorlarım artmış durumda.
