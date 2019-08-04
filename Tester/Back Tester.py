import pandas as pd
from features import *
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Name.csv')
data.columns = ['date','open','high','low','close','volume']
data = data.set_index(pd.to_datetime(data.date))
data = data[['open','high','low','close','volume']]
prices = data.drop_duplicates(keep=False)

momentumKey = [3,4,5,8,9,10] 
stochasticKey = [3,4,5,8,9,10] 
williamsKey = [6,7,8,9,10] 
procKey = [12,13,14,15] 
wadlKey = [15] 
adoscKey = [2,3,4,5] 
macdKey = [15,30] 
cciKey = [15] 
bollingerKey = [15] 
heikenashiKey = [15] 
paverageKey = [2] 
slopeKey = [3,4,5,10,20,30] 
fourierKey = [10,20,30] 
sineKey = [5,6] 


keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey, heikenashiKey
           ,paverageKey,slopeKey,fourierKey,sineKey] 

momentumDict = momentum(prices,momentumKey) 
print('1') 
stochasticDict = stochastic(prices,stochasticKey) 
print('2') 
williamsDict = williams(prices,williamsKey)
print('3') 
procDict = proc(prices,procKey) 
print('4') 
wadlDict = wadl(prices,wadlKey) 
print('5')
adoscDict = adosc(prices,adoscKey)
print('6') 
macdDict = macd(prices,macdKey) 
print('7') 
cciDict = cci(prices,cciKey) 
print('8')
bollingerDict = bollinger(prices,bollingerKey,2) 
print('9') 

hkaprices = prices.copy()
hkaprices['Symbol']='SYMB'
HKA = OHLCresample(hkaprices,'15H')

heikenDict = Heiken_Ashi(HKA,heikenashiKey) 
print('10' ) 
paverageDict = pavarage(prices,paverageKey) 
print('11') 
slopeDict = slopes(prices,slopeKey) 
print('12') 
fourierDict = fourier(prices,fourierKey) 
print('13') 
sineDict = sine(prices,sineKey) 
print('14') 

# Create list of dictionaries 

dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
            ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
            ,cciDict.cci,bollingerDict.bands,heikenDict.candles,paverageDict.avs
            ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs] 

# list of column name on csv

colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
           'cci','bollinger','heiken','paverage','slope','fourier','sine']

masterFrame = pd.DataFrame(index = prices.index) 
for i in range(0,len(dictlist)): 
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1]) 
        masterFrame[colID] = dictlist[i]
    else: 
        for j in keylist[i]: 
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + str(k)
                masterFrame[colID] = dictlist[i][j][k]

threshold = round(0.7*len(masterFrame)) 
masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]

masterFrame.heiken150 = masterFrame.heiken150.fillna(method='bfill') 
masterFrame.heiken151 = masterFrame.heiken151.fillna(method='bfill')
masterFrame.heiken152 = masterFrame.heiken152.fillna(method='bfill')
masterFrame.heiken153 = masterFrame.heiken153.fillna(method='bfill') 

# Drop columns that have 30% or more NAN data 

masterFrameCleaned = masterFrame.copy() 

masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
masterFrameCleaned.to_csv('calculated_BT.csv')
print('completed')

columns = ['momentum3close','momentum4close'
,'momentum5close','momentum8close','momentum9close','momentum10close'
,'stoch3K','stoch3D','stoch4K','stoch4D'
,'stoch5K','stoch5D','stoch8K','stoch8D'
,'stoch9K','stoch9D','stoch10K'
,'stoch10D','will6R','will7R','will8R'
,'will9R','will10R','proc12close','proc13close'
,'proc14close','proc15close','wadl15close','adosc2AD'
,'adosc3AD','adosc4AD','adosc5AD','macd1530','cci15close'
,'bollinger15upper','bollinger15mid','bollinger15lower','heiken150'
,'heiken151','heiken152','heiken153','paverage2open'
,'paverage2high','paverage2low','paverage2close','slope3high','slope4high','slope5high'
,'slope10high','slope20high','slope30high'
,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']


clf_load_dt = joblib.load('saved_model_dt.pkl')
clf_load_knn = joblib.load('saved_model_knn.pkl')
clf_load_lr = joblib.load('saved_model_lr.pkl')
clf_load_nn = joblib.load('saved_model_nn.pkl')
clf_load_rf = joblib.load('saved_model_rf.pkl')
clf_load_svm = joblib.load('saved_model_svm.pkl')

#------------------------------------------------------------------------------
save_state_old = 3
save_state_new = 3
save_price = 0
sum_profit = 0
first = 1
maxx=0
minn=0
c=0
s=0
x=1


df = masterFrameCleaned[list(columns)].values
a = []

std = StandardScaler()
newfeaturess = std.fit_transform(df)

for i in range(0,len(masterFrameCleaned)):
    profit = 0
    a=[]
    newfeatures = newfeaturess[i].reshape(1, 73) 

    predicted_dt = clf_load_dt.predict(newfeatures)
    a = np.append(a,predicted_dt)
    
    predicted_rf = clf_load_rf.predict(newfeatures)
    a = np.append(a,predicted_rf)
    
    predicted_knn = clf_load_knn.predict(newfeatures)
    a = np.append(a,predicted_knn)
    
    predicted_lr = clf_load_lr.predict(newfeatures)
    a = np.append(a,predicted_lr)
    
    predicted_nn = clf_load_nn.predict(newfeatures)
    a = np.append(a,predicted_nn)
    
    predicted_svm = clf_load_svm.predict(newfeatures)
    a = np.append(a,predicted_svm)

    buy = 0
    sell = 0
    for ii in a:
        if ii == 1:
            buy = buy + 1
        elif ii == 0:
            sell = sell + 1

#------------------------------------------------------------------------------
#calculate profit
    if first == 1:
        if buy > 3:#buy
            first=2
            save_state_new = 1
            save_price = masterFrameCleaned.close.iloc[i]
        elif sell > 3:#sell
            first=2
            save_state_new = 0
            save_price = masterFrameCleaned.close.iloc[i]
        save_state_old = save_state_new
        
    if first == 2:
        if buy > 3:#buy
            save_state_new = 1
        elif sell > 3:#sell
            save_state_new = 0
        else:#hold
            save_state_new = 3
    
    if save_state_old != save_state_new:
        if save_state_old == 1:
            profit = masterFrameCleaned.close.iloc[i] - save_price - 0.0002
            if profit > 0:
                c=c+1
            elif profit < 0:
                s=s+1            
            save_price = masterFrameCleaned.close.iloc[i]
        elif save_state_old == 0:
            profit = save_price - masterFrameCleaned.close.iloc[i] - 0.0002
            if profit > 0:
                c=c+1
            elif profit < 0:
                s=s+1
            save_price = masterFrameCleaned.close.iloc[i]
        elif save_state_old == 3:
            save_price = masterFrameCleaned.close.iloc[i]
            
        if maxx < profit: 
            maxx = profit
        if minn > profit: 
            minn = profit 
        save_state_old = save_state_new

    sum_profit = sum_profit + profit

#------------------------------------------------------------------------------
  
if save_state_old == save_state_new:
    if save_state_old == 1:
        profit = masterFrameCleaned.close.iloc[i] - save_price - 0.0002
        if profit > 0:
            c=c+1
        elif profit < 0:
            s=s+1            
    elif save_state_old == 0:
        profit = save_price - masterFrameCleaned.close.iloc[i] - 0.0002
        if profit > 0:
            c=c+1
        elif profit < 0:
            s=s+1
                    
    if maxx < profit: 
        maxx = profit
    if minn > profit: 
        minn = profit 
    sum_profit = sum_profit + profit


print('profit of 10 lots:',sum_profit*10000)
print('Total number of trades:',s+c)
print('sum of wining trades:',c)
print('sum of loss trades:',s)
print('max Down for 10 lots:',minn*10000)
print('max up for 10 lots:',maxx*10000)


#------------------------------------------------------------------------------










