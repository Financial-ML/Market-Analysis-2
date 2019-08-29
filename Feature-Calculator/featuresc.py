import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#------------------------------------------------------------------------------
class Holder:
    1
#------------------------------------------------------------------------------

def Heiken_Ashi(price,period):
    result = Holder()
    dict = {}
    
    HAclose = price[['open','high','low','close']].sum(axis=1)/4
    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]
    HAhigh = HAclose.copy()
    HAlow = HAclose.copy()  
    
    for i in range(1,len(price)):
        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([price.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i] = np.array([price.low.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).min()
    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.coulmns = ['open','high','low','close']
    
    df.index = df.index.droplevel(0)

    dict[period[0]]=df
    result.candles=dict
    return result
#------------------------------------------------------------------------------
def detrend(price,method='difference'):
    if method=='difference':
        detrended=price.close[1:]-price.close[:-1].values
    elif method=='linear':
        x = np.arange(0,len(price))
        y = price.close.values
        model = LinearRegression()
        model.fit(x.reshape(-1,1),y.reshape(-1,1))
        trend = model.predict(x.reshape(-1,1))
        trend = trend.reshape((len(price),))
        detrended = price.close - trend
    else:
        print("error trend")
    return detrended
#------------------------------------------------------------------------------
def fseries(x,a0,a1,b1,w):
    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)
    return f
#------------------------------------------------------------------------------
def sseries(x,a0,b1,w):
    f = a0 + b1 * np.sin(w * x)
    return f
#------------------------------------------------------------------------------
def fourier(price,period,method='difference'):
    results = Holder()
    dict = {}
    plot = False
    detrended = detrend(price,method)
    
    for i in range(0,len(period)):
        coeffs =[]
        for j in range(period[i],len(price)):
            x = np.arange(0,period[i])
            y = detrended.iloc[j-period[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                
                try:
                    res = scipy.optimize.curve_fit(fseries,x,y)
                    
                except (RuntimeError,OptimizeWarning):
                    res = np.empty((1,4))
                    res[0,:] = np.NAN
                
            if plot == True:
                xt = np.linspace(0,period[i],100)
                yt = fseries(xt,res[0][0],res[0][1],res[0][2],res[0][3])
                
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
                
            coeffs = np.append(coeffs,res[0],axis=0)
            
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs).reshape(((len(coeffs)//4,4)))
        df = pd.DataFrame(coeffs,index=price.iloc[period[i]:len(price)].index)
        df.columns = ['a0','a1','b1','w']
        df = df.fillna(method='bfill')
        dict[period[i]] = df
    results.coeffs = dict
    return results
#------------------------------------------------------------------------------
def sine(price,period,method='difference'):
    results = Holder()
    dict = {}
    plot = False
    detrended = detrend(price,method)
    
    for i in range(0,len(period)):
        coeffs =[]
        for j in range(period[i],len(price)):
            x = np.arange(0,period[i])
            y = detrended.iloc[j-period[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                
                try:
                    res = scipy.optimize.curve_fit(sseries,x,y)
                    
                except (RuntimeError,OptimizeWarning):
                    res = np.empty((1,3))
                    res[0,:] = np.NAN
                
            if plot == True:
                xt = np.linspace(0,period[i],100)
                yt = sseries(xt,res[0][0],res[0][1],res[0][2])
                
                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
                
            coeffs = np.append(coeffs,res[0],axis=0)
            
        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs).reshape(((len(coeffs)//3,3)))
        df = pd.DataFrame(coeffs,index=price.iloc[period[i]:len(price)].index)
        df.columns = ['a0','b1','w']
        df = df.fillna(method='bfill')
        dict[period[i]]=df
    results.coeffs = dict
    return results
#------------------------------------------------------------------------------
def wadl(price,period):
    
    results = Holder()
    dict = {}
    
    for i in range(0,len(period)):
        
        WAD = []
        
        for j in range(period[i],len(price)):
            
            TRH = np.array([price.high.iloc[j],price.close.iloc[j-1]]).max()
            TRL = np.array([price.low.iloc[j],price.close.iloc[j-1]]).min()
            
            if price.close.iloc[j] > price.close.iloc[j-1]:
                PM = price.close.iloc[j]-TRL
            elif price.close.iloc[j] < price.close.iloc[j-1]:
                PM = price.close.iloc[j]-TRH
            elif price.close.iloc[j] == price.close.iloc[j-1]:
                PM = 0
            else:
                print("error in wadl")
                
            AD = PM * price.volume.iloc[j]
            WAD = np.append(WAD,AD)
            
        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD,index=price.iloc[period[i]:len(price)].index)
        WAD.columns = ['close']
        dict[period[i]] = WAD
        
    results.wadl = dict
    return results
#------------------------------------------------------------------------------

def OHLCresample(DataFrame,TimeFrame,column='ask'):
    
    grouped = DataFrame.groupby('Symbol')    
    
    if np.any(DataFrame.columns == 'Ask'): 
        
        if column =='ask':
            ask = grouped['Ask'].resample(TimeFrame).ohlc()
            askVol = grouped['AskVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = askVol
            
        elif column =='bid':
            bid = grouped['Bid'].resample(TimeFrame).ohlc()
            bidVol = grouped['BidVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bidVol
            
        else:
            raise ValueError('error OHLCresample')
            
    elif np.any(DataFrame.columns == 'close'):
        open = grouped['open'].resample(TimeFrame).ohlc()
        close = grouped['close'].resample(TimeFrame).ohlc()
        high = grouped['high'].resample(TimeFrame).ohlc()
        low = grouped['low'].resample(TimeFrame).ohlc()
        askVol = grouped['volume'].resample(TimeFrame).count() 
    
        resampled = pd.DataFrame(open) 
        resampled['high'] = high 
        resampled['low'] = low 
        resampled['close'] = close 
        resampled['volume'] = askVol 
             

    resampled = resampled.dropna()

    return resampled
#------------------------------------------------------------------------------
def momentum(prices,periods):
    
    results = Holder() 
    open = {} 
    close = {} 
    for i in range(0,len(periods)): 
        
        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index) 
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index) 
        
        open[periods[i]].columns = ['open']
        close[periods[i]].columns = ['close'] 
    
    results.open = open 
    results.close = close 
    
    return results 

#------------------------------------------------------------------------------

def stochastic(prices,periods):
    results = Holder() 
    close = {} 
    
    for i in range(0,len(periods)):
        Ks=[]
        for j in range(periods[i],len(prices)):
            C= prices.close.iloc[j]
            H = prices.high.iloc[j-periods[i]:j-1].max() 
            L= prices.low.iloc[j-periods[i]:j-1].min() 
            if H == L:
                K = 0 
            else: 
                K= 100*(C-L)/(H-L) 
            Ks = np.append(Ks,K)
            
        df = pd.DataFrame(Ks,index =prices.iloc[periods[i]:len(prices)].index) 
        df.columns = ['K']
        df['D'] = df.K.rolling(3).mean() 
        df = df.dropna() 
        close[periods[i]] = df 
    results.close = close 
        
    return results 

#------------------------------------------------------------------------------

def williams(prices,periods):
    results = Holder() 
    close = {} 
    for i in range(0,len(periods)):
        Rs=[]
        for j in range(periods[i],len(prices)):
            
            C= prices.close.iloc[j] 
            H = prices.high.iloc[j-periods[i]:j-1].max() 
            L= prices.low.iloc[j-periods[i]:j-1].min() 
            if H == L:
                R = 0 
            else: 
                R= 100*(H-C)/(H-L) 
            Rs = np.append(Rs,R)
            
        df = pd.DataFrame(Rs,index = prices.iloc[periods[i]:len(prices)].index) 
        df.columns = ['R']
        df = df.dropna() 
        close[periods[i]] = df 
        
    results.close = close 
        
    return results 

#------------------------------------------------------------------------------

def proc(prices,periods):
    results = Holder() 
    proc = {} 
    for i in range(0,len(periods)):
        proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values)/prices.close.iloc[:-periods[i]].values)
        proc[periods[i]].columns = ['close']
    
    results.proc = proc
    return results

#------------------------------------------------------------------------------

def adosc(prices,periods):
    results = Holder() 
    accdist = {} 
    for i in range(0,len(periods)):
        AD=[]
        for j in range(periods[i],len(prices)):
            
            C = prices.close.iloc[j] 
            H = prices.high.iloc[j-periods[i]:j-1].max() 
            L = prices.low.iloc[j-periods[i]:j-1].min() 
            V = prices.volume.iloc[j]
            if H == L:
                CLV = 0 
            else: 
                CLV= ((C-L)-(H-C))/(H-L) 
            AD = np.append(AD,CLV*V)
        AD = AD.cumsum()   
        AD = pd.DataFrame(AD,index =prices.iloc[periods[i]:len(prices)].index) 
        AD.columns = ['AD']
        accdist[periods[i]] = AD 
        
    results.AD = accdist 
        
    return results 

#------------------------------------------------------------------------------

def macd(prices,periods):
    
    results = Holder() 
    
    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm( span=periods[1]).mean()
    
    MACD = pd.DataFrame(EMA1 - EMA2)
    MACD.columns = ['L'] 

    sigMACD = MACD.rolling(3).mean() 
    sigMACD.columns = ['SL'] 
    
    results.line = MACD 
    results.signal = sigMACD 

    return results

#------------------------------------------------------------------------------
    
def cci (prices,periods):
    results = Holder() 
    CCI = {}

    for i in range(0,len(periods)): 
        MA = prices.close.rolling(periods[i]).mean() 
        std = prices.close.rolling(periods[i]).std() 

        D = (prices.close-MA)/std 
        CCI[periods[i]] = pd.DataFrame((prices.close-MA)/(0.015*D)) 
        CCI[periods[i]].columns = ['close'] 

    results.cci = CCI 
    
    return results

#------------------------------------------------------------------------------

def bollinger (prices,periods,deviations):
    results = Holder() 
    boll = {}
    for i in range(0,len(periods)): 
        mid = prices.close.rolling(periods[i]).mean() 
        std = prices.close.rolling(periods[i]).std() 

        upper = mid+deviations*std
        lower = mid-deviations*std
        
        df = pd.concat((upper,mid,lower),axis=1)
        df.columns = ['upper','mid','lower']
        
        boll[periods[i]] = df
        
    results.bands = boll
    return results

#------------------------------------------------------------------------------
    
def pavarage (prices,periods):
    results = Holder() 
    avs = {}
    for i in range(0,len(periods)): 
        avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())
        
    results.avs = avs
    return results

#------------------------------------------------------------------------------

def slopes (prices,periods):
    results = Holder() 
    slope = {}
    
    for i in range(0,len(periods)):
        ms=[]
        for j in range(periods[i],len(prices)):
            y = prices.high.iloc[j-periods[i]:j].values
            x = np.arange(0,len(y))
            
            res = stats.linregress(x,y=y)
            m = res.slope
            ms = np.append(ms,m)
            
        
        ms = pd.DataFrame(ms,index=prices.iloc[periods[i]:len(prices)].index)
        ms.columns = ['high']
        slope[periods[i]] = ms 
        
        
    results.slope = slope 
    
    return results

#------------------------------------------------------------------------------
    
def Market (prices,periods):
    results = Holder() 
    slope = {}
    m=3
    prices.index = pd.to_datetime(prices.index)
    prices['B'] = prices.index.minute
    
    for i in range(0,len(periods)):
        ms=[]
        for j in range(periods[i],len(prices)-60):
            if prices.B.iloc[j]==0:
                if prices.open[j] < prices.close[j+60]:
                    m=1
                elif prices.open[j] >= prices.close[j+60]:
                    m=0
            ms.append(m)
        ms = pd.DataFrame(ms,index=prices.iloc[periods[i]:len(prices)-60].index)
        ms.columns = ['Market']
        slope[periods[i]] = ms 
        
        
    results.slope = slope 
    
    return results

#------------------------------------------------------------------------------
def resamble(prices):
    results = Holder() 
    maxHigh=0
    maxLow=0
    msOpen=[]
    msHigh=[]
    msLow=[]
    msClose=[]
    msVolume=[]
    prices.index = pd.to_datetime(prices.index)
    prices['B'] = prices.index.minute
    Open=prices.open.iloc[0]
    High=prices.high.iloc[0]
    Low=prices.low.iloc[0]
    Close=prices.close.iloc[0]
    Volume=prices.volume.iloc[0]
    
    for i in range(0,len(prices)):
        x=0
        if prices.B.iloc[i]==0:
            Openn=prices.open.iloc[i]
            maxHigh=prices.open.iloc[i]
            maxLow=prices.open.iloc[i]
            Volume=prices.volume.iloc[i]
            x=1
        if maxHigh<prices.high.iloc[i]:
            maxHigh=prices.high.iloc[i]
        if maxLow>prices.low.iloc[i]:
            maxLow=prices.low.iloc[i]
            
        Open = Openn
        High=maxHigh
        Low=maxLow
        Close=prices.close.iloc[i]
        if x!=1:
            Volume=Volume+prices.volume.iloc[i]
        
        msOpen=np.append(msOpen,Open)
        msHigh=np.append(msHigh,High)
        msLow=np.append(msLow,Low)
        msClose=np.append(msClose,Close)
        msVolume=np.append(msVolume,Volume)
        
    df1 = pd.DataFrame(msOpen,index=prices.iloc[0:len(prices)].index)
    df1.columns = ['open']
    df2 = pd.DataFrame(msHigh,index=prices.iloc[0:len(prices)].index)
    df2.columns = ['high']
    df3 = pd.DataFrame(msLow,index=prices.iloc[0:len(prices)].index)
    df3.columns = ['low']
    df4 = pd.DataFrame(msClose,index=prices.iloc[0:len(prices)].index)
    df4.columns = ['close']
    df5 = pd.DataFrame(msVolume,index=prices.iloc[0:len(prices)].index)
    df5.columns = ['volume']
    result = pd.concat([df1, df2, df3, df4, df5], axis=1, sort=False)
    
    results = result
    
    return results

#------------------------------------------------------------------------------

    
    
    
    
    
    
    
    
    
    
    
    
        