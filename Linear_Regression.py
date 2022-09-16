######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/advertising.csv")
df.shape #Out[3]: (200, 4)


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

X=df[["TV"]]
y=df[["sales"]]



##########################
# Model
##########################

reg_model=LinearRegression().fit(X,y)



# y_hat = b + w*TV

# sabit (b - bias)

reg_model.intercept_[0] # b= 7.032593549127693



# tv'nin katsayısı (w1)
reg_model.coef_[0][0] # xin katsayısı =0.04753664




##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 150 #Out[16]: 14.163089614080658

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 500 #Out[17]: 30.800913765637574

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()



##########################
# Tahmin Başarısı
##########################

# MSE

y_pred=reg_model.predict(X)
mean_squared_error(y,y_pred)

#Out[20]: 10.512652915656757

# RMSE
np.sqrt(mean_squared_error(y,y_pred))

#Out[21]: 3.2423221486546887

#R-KARE

reg_model.score(X,y)  #Out[23]: 0.611875050850071


######################################################
# Multiple Linear Regression
######################################################


df=pd.read_csv("MIUUL/MACHİNE_LEARNING_/advertising.csv")

y = df[["sales"]]
X=df.drop("sales",axis=1)

##########################
# Model
##########################

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

y_train.shape    y_test.shape
X_train.shape    X_test.shape


reg_model=LinearRegression().fit(X_train,y_train)

#sabit

reg_model.intercept_[0] #Out[35]: 2.9079470208164295
reg_model.coef_[0][0] #Out[36]: 0.04684310317699043

##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40
reg_model.intercept_[0] + reg_model.coef_[0][0]*30 + reg_model.coef_[0][0]*10 + reg_model.coef_[0][0]*40


#programatikleştirelim.

yeni_veri=[[30],[10],[40]]

yeni_veri=pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)




##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE

y_pred=reg_model.predict(X_train)
mean_squared_error(y_train,y_pred)

#Out[44]: 3.0168306076596774

#TRAIN RKARE

reg_model.score(X_train,y_train)

#Out[46]: 0.8959372632325174


# Test RMSE

y_pred=reg_model.predict(X_test)
mean_squared_error(y_test,y_pred)

#Out[45]: 1.9918855518287906

#TEST RKARE

reg_model.score(X_test,y_test)

#Out[47]: 0.8927605914615384



# 10 Katlı CV RMSE


np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_squared_error")))

#Out[48]: 1.6913531708051799


# 5 Katlı CV RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=5,scoring="neg_mean_squared_error")))

#Out[49]: 1.7175247278732086



######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("MIUUL/MACHİNE_LEARNING_/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

















