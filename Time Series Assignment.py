"""
Time Series: Stock and Watson Vector Autoregressions 2001 
Author: Omer Abdelrahim
"""
#%% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#%% Data Wrangling 
#Parts 1-2
dfini=pd.read_csv(r"E:\Downloads\fredgraph.csv")
dfini.dtypes
dfini['DATE']=pd.to_datetime(dfini['DATE'])
dfmid=dfini.set_index('DATE')
dfmid.drop('2023-10-01',inplace=True,axis=0)
dfmid.dtypes
df=dfmid.astype('float64')
df.dtypes
df['IRATE']=400*np.log(df['GDPCTPI']/df['GDPCTPI'].shift(1))
df.dropna(inplace=True)
df.head()

#%% Testing and Training data 
#Part 3
df_train=df[:'2009-10-01']
df_test=df['2010-01-01':]

#%% Plot PACF 
# Part 4
plot_pacf(df_train['IRATE'],lags=20,title='1960-2000 Inflation Rate')
plt.grid();

plot_pacf(df_train['UNRATE'],lags=20,title='1960-2000 Unemployment Rate')
plt.grid();

plot_pacf(df_train['DFF'],lags=20,title='1960-2000 Federal Funds Rate')
plt.grid();

# Insert some observations on the important lags 

#%% Optimal number of lags 
# Part 5 
print(f'Optimal lags for the Inflation Rate using AIC are:{ar_select_order(df_train["IRATE"],maxlag=20,ic="aic").ar_lags}')
print(f'Optimal lags for the Unemployment Rate using AIC are:{ar_select_order(df_train["UNRATE"],maxlag=20,ic="aic").ar_lags}')
print(f'Optimal lags for the Federal Funds Rate using AIC are:{ar_select_order(df_train["DFF"],maxlag=20,ic="aic").ar_lags}')

# Answer question about important lags vs what I saw in part 4 

#%% Forecasting for Testing Data 
# Part 6 
# Inflation Rate 
model = AutoReg(df_train['IRATE'],lags=3)
model_fit=model.fit()
forecast = model_fit.predict(start='2010',end='2023')
model_fit.plot_predict(start='2010',end='2023');
plt.plot(df['IRATE']['1960':'2023']);

# Unemployment Rate
model1 = AutoReg(df_train['UNRATE'],lags=2)
model1_fit=model1.fit()
forecast1 = model1_fit.predict(start='2010',end='2023')
model1_fit.plot_predict(start='2010',end='2023');
plt.plot(df['UNRATE']['1960':'2023']);

# Federal Funds Rate 
model2 = AutoReg(df_train['DFF'],lags=8)
model2_fit=model2.fit() 
forecast2 = model2_fit.predict(start='2010',end='2023')
model2_fit.plot_predict(start='2010',end='2023');
plt.plot(df['DFF']['1960':'2023']);
#%% Extended Forecast 
# Part 7 
# Inflation Rate
nmodel = AutoReg(df['IRATE'],lags=3)
nmodel_fit=nmodel.fit()
fforecast = nmodel_fit.predict(start='2023',end='2025')
nmodel_fit.plot_predict(start='2023',end='2025');
plt.plot(df['IRATE']['2000':'2025']);

# Unemployment Rate 
nmodel1 = AutoReg(df['UNRATE'],lags=2)
nmodel1_fit=nmodel1.fit()
fforecast1 = nmodel1_fit.predict(start='2023',end='2025')
nmodel1_fit.plot_predict(start='2023',end='2025');
plt.plot(df['UNRATE']['2000':'2025']);

# Federal Funds Rate 
nmodel2 = AutoReg(df['DFF'],lags=8)
nmodel2_fit=nmodel2.fit()
fforecast2 = nmodel2_fit.predict(start='2023',end='2025')
nmodel2_fit.plot_predict(start='2023',end='2025');
plt.plot(df['DFF']['2000':'2025']);
#%% VAR Model 
#Part 8
ifirst=df_train[['IRATE','UNRATE','DFF']]
ifirst.head()
vmod = VAR(ifirst)
results = vmod.fit(4)
irf = results.irf(24)
fig = irf.plot(orth=True,signif=0.1);

#%% FEVD 
# Part 9 
fevd = results.fevd(12)
fevd.summary()
fevd.plot();

#%% Granger Causality 
# Part 10 
print(results.test_causality(['UNRATE'], ['IRATE'], kind='f').summary())
print(results.test_causality(['IRATE'], ['UNRATE'], kind='f').summary())

print(results.test_causality(['UNRATE'], ['DFF'], kind='f').summary())
print(results.test_causality(['DFF'], ['UNRATE'], kind='f').summary())

print(results.test_causality(['IRATE'], ['DFF'], kind='f').summary())
print(results.test_causality(['DFF'], ['IRATE'], kind='f').summary())

#%% Full Sample Models 
# Part 11 (7-10)
# Part 8.2 
ifirst1=df[['IRATE','UNRATE','DFF']]
vmod1 = VAR(ifirst1)
results1 = vmod1.fit(4)
irf1 = results1.irf(24)
fig1 = irf1.plot(orth=True,signif=0.1);

#Part 9.2 
fevd1 = results1.fevd(12)
fevd1.summary()
fevd1.plot();

#Part 10.2 
print(results1.test_causality(['UNRATE'], ['IRATE'], kind='f').summary())
print(results1.test_causality(['IRATE'], ['UNRATE'], kind='f').summary())

print(results1.test_causality(['UNRATE'], ['DFF'], kind='f').summary())
print(results1.test_causality(['DFF'], ['UNRATE'], kind='f').summary())

print(results1.test_causality(['IRATE'], ['DFF'], kind='f').summary())
print(results1.test_causality(['DFF'], ['IRATE'], kind='f').summary())
