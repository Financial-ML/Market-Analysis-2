import pandas as pd
from featuresc import *
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import fxcmpy
import time

#------------------------------------------------------------------------------

try:
    TOKEN = "2fab528ed0c184c625ee0d1970058a1651012b93"
            
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
    
    if con.is_connected() == True:
        print("Data retrieved...")
        df = con.get_candles('EUR/USD', period='m5',number=200)
        df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
                
        df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
        df = df[['open','high','low','close','volume']]
        df = df[~df.index.duplicated()]
        prices = df.copy()
    else:
        print('No connection with fxcm')
    

#------------------------------------------------------------------------------
    prices.index=pd.to_datetime(prices.index)
    prices['B'] = prices.index.minute
    for i in range(0,len(prices)):
        if prices.B.iloc[i] == 0:
            break
    prices = prices[i:]
    prices.drop(['B'], axis=1)
    
#------------------------------------------------------------------------------
            
    momentumKey = [3,4,5,8,9,10] 
    stochasticKey = [3,4,5,8,9,10] 
    williamsKey = [6,7,8,9,10] 
    procKey = [12,13,14,15] 
    wadlKey = [15] 
    adoscKey = [2,3,4,5] 
    macdKey = [15,30] 
    cciKey = [15] 
    bollingerKey = [15] 
    paverageKey = [2] 
    slopeKey = [3,4,5,10,20,30] 
    fourierKey = [10,20,30] 
    sineKey = [5,6] 
            
            
    keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
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
                ,cciDict.cci,bollingerDict.bands,paverageDict.avs
                ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs] 
            
        # list of column name on csv
            
    colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
               'cci','bollinger','paverage','slope','fourier','sine']
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
    
            # Drop columns that have 30% or more NAN data 
            
    masterFrameCleaned = masterFrame.copy() 
            
    masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
    masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
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
    ,'bollinger15upper','bollinger15mid','bollinger15lower','paverage2open'
    ,'paverage2high','paverage2low','paverage2close','slope3high','slope4high','slope5high'
    ,'slope10high','slope20high','slope30high'
    ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0','fourier20a1','fourier20b1','fourier20w','fourier30a0'
    ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w','sine6a0','sine6b1','sine6w','open','high','low','close']
            
            
    clf_load_dt = joblib.load('saved_model_dt.pkl')
    clf_load_rf = joblib.load('saved_model_rf.pkl')
            
            
            
    df = masterFrameCleaned[list(columns)].values
    a = []
    std = StandardScaler()
    newfeaturess = std.fit_transform(df)
            
    a=[]
    a1=[]
    newfeatures = newfeaturess[-1].reshape(1, 69) 
        
    predicted_dt = clf_load_dt.predict(newfeatures)
    a = np.append(a,predicted_dt)
                
    predicted_rf = clf_load_rf.predict(newfeatures)
    a = np.append(a,predicted_rf)
            
    newfeatures = newfeaturess[-2].reshape(1, 69)
    predicted_dt = clf_load_dt.predict(newfeatures)
    a1 = np.append(a1,predicted_dt)
        
    predicted_rf = clf_load_rf.predict(newfeatures)
    a1 = np.append(a1,predicted_rf)
        
    print(a)
    print(a1)
            
    if con.is_connected() == True:
        if a[0]==1 and a[1]==1 and a1[0]==1 and a1[1]==1:
            print("buy")
            if con.open_pos == {}:
                con.create_market_buy_order('EUR/USD', 10)
            else:
                tradeId = con.get_open_trade_ids()[0]
                pos = con.get_open_position(tradeId)
                if pos.get_isBuy()==False:
                    con.close_all_for_symbol('EUR/USD')
                    time.sleep(5)
                    con.create_market_buy_order('EUR/USD', 10)
                #see if there is no order enter one
                #if there is order buy keep it else complet
                #close sell order
                #open order buy
        if a[0]==0 and a[1]==0 and a1[0]==0 and a1[1]==0:
            print("sell")
            if con.open_pos == {}:
                con.create_market_sell_order('EUR/USD', 10)
            else:
                tradeId = con.get_open_trade_ids()[0]
                pos = con.get_open_position(tradeId)
                if pos.get_isBuy()==True:
                    con.close_all_for_symbol('EUR/USD')
                    time.sleep(5)
                    con.create_market_sell_order('EUR/USD', 10)
                #see if there is order
                #see the type if sell keep it else complet
                #close order buy
                #open order sell
    else:
        print('No connection with fxcm')   
        
except Exception as e:
    
    print("code does not work:",e)
    
    if con.is_connected() == True:
        con.close_all_for_symbol('EUR/USD')
    else:
        print('No connection with fxcm')
#------------------------------------------------------------------------------


