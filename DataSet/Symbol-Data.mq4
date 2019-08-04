//+------------------------------------------------------------------+
//|                                                  Symbol-Data.mq4 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
int num_bar = 10000;
long volume;
double open,close,high,low;
datetime date;
int filehandle=FileOpen("Name_Of_File.csv",FILE_WRITE|FILE_READ|FILE_CSV,',');
   if(filehandle!=INVALID_HANDLE)
     {
      FileWrite(filehandle,"date","open","high","low","close","volume");
      for(int i=num_bar;i>10;i--){
      date=iTime(NULL,0,i);
      open=Open[i];
      high=High[i];
      low=Low[i];
      close=Close[i];
      volume=Volume[i];
      FileWrite(filehandle,date,open,high,low,close,volume);
      }
      FileClose(filehandle);
     }
   else Print("Operation FileOpen failed, error ",GetLastError());
  }
//+------------------------------------------------------------------+
