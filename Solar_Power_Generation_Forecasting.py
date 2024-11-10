import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

data=pd.read_excel("Pv.xlsx")
print(data.head(15)) 

#Creating a Correlation Graph
cor_mat = np.corrcoef(data.T)
print ("Correlation matrisinin sekli:{}".format(cor_mat.shape))
plt.subplot(2,1,1)
sns.heatmap(cor_mat, 
    xticklabels=data.columns, 
    yticklabels=data.columns)
plt.show()

#Separation into training and test datasets
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=0)  # eğitim ve test veri setini %70 ve %30 olacak şekilde ayırdık

#Descriptive statistical information of independent variables
print("\n")
print("Descriptive statistical information of independent variables")
print("----------------------------------------------------------------\n")
print(stats.describe(X_train, axis=0))
print(np.std(X_train, axis=0))
print(stats.describe(X_test, axis=0))
print(np.std(X_test, axis=0))

#Descriptive statistical information of the dependent variable
print("\n")
print("Descriptive statistical information of the dependent variable")
print("---------------------------------------------------------------\n")
print(stats.describe(y_train, axis=0))
print(np.std(y_train, axis=0))
print(stats.describe(y_test, axis=0))
print(np.std(y_test, axis=0))

#Attribute scaling process (data normalisation process)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1))
y_test=scaler.fit_transform(y_test.reshape(-1,1))

#Drawing the box plot of the data set
dt=pd.DataFrame(scaler.fit_transform(data))
dt.columns=data.columns
plt.subplot(2,1,2)
plt.boxplot(dt, labels=dt.columns)
plt.show()


#Fitting the training set according to multiple linear regression
from sklearn.linear_model import LinearRegression
model_Regresyon=LinearRegression()
model_Regresyon.fit(X_train, y_train)
forecast=model_Regresyon.predict(X_test)

#Selection of effective independent variables by backward elimination technique
import statsmodels.api as sm
X=np.append(arr=np.ones((1620,1)).astype(int), values=X, axis=1)
X_new=X[:,[0,1,2,3,4,5,6,7]]
model_Regresyon_OLS=sm.OLS(endog=y, exog=X_new).fit()
print(model_Regresyon_OLS.summary())

X_opt=X[:,[0,1,2,3,4,5,6,7]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

#Performance Measures of the test data set
print("\n-----------------------------------------------------\n")
print("Performance Measures of the test data set")
print("MAE={:.4f}".format(mean_absolute_error(y_test, forecast)))
print("MSE={:.4f}".format(mean_squared_error(y_test, forecast)))
print("MedAE= {:.4f}".format(median_absolute_error(y_test, forecast)))
print("Belirleme Katsayısı(R^2)={:.4f}".format(r2_score(y_test, forecast)))
print("RMSE= {:.4f}".format(np.sqrt(mean_squared_error(y_test, forecast))))
print("MAPE= {:.4f}".format(mean_absolute_percentage_error(y_test, forecast)*100))

