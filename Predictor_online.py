from keras.engine import training
import matplotlib as plt
import math
import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , LSTM
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pandas_datareader as web
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
plt.style.use('dark_background')

st.write('''
# MOM Finance
** Trader bot **

''')
img=Image.open("d:/IT/Pyton/Exercise 2/1218.jpg")
st.image(img,width=600)

st.sidebar.header(' INSERT DATA ')


symbol  = st.sidebar.selectbox('Select the symbol : ' , ['AAPL' , 'AMZN' , 'TSLA' , 'NFLX' , 'BTC' , 'ETH' , 'BNB'])

n = 1


aapl = web.DataReader('AAPL' , 'yahoo' , start='2010-01-01' , end= 'today' )
amzn = web.DataReader('AMZN' , 'yahoo' , start='2019-01-01' , end='today' )
tsla = web.DataReader('TSLA' , 'yahoo' , start='2010-01-01' , end= 'today' )
nflx = web.DataReader('NFLX' , 'yahoo' , start='2010-01-01' , end= 'today' )
btc = web.DataReader('BTC-USD' , 'yahoo' , start='2010-01-01' , end= 'today' )
eth = web.DataReader('ETH-usd' , 'yahoo' , start='2010-01-01' , end= 'today' )
bnb = web.DataReader('BNB-USD' , 'yahoo' , start='2019-01-01' , end='today' )



def get_data(symbol):
    if symbol=='AAPL' :
        return aapl
    elif symbol == 'AMZN' :
        return amzn
    elif symbol == 'TSLA' :
        return tsla
    elif symbol == 'NFLX' :
        return nflx
    elif symbol == 'BTC' :
        return btc
    elif symbol == 'ETH' :
        return eth
    elif symbol == 'BNB' :
       return bnb 

    # df=df.set_index(pd.DatetimeIndex(df['Date'].values))
    # return df

def get_company_name(symbol):
    if symbol == 'AAPL' :
        return 'AAPL'
    elif symbol == 'AMZN' :
        return 'AMZN'
    elif symbol == 'TSLA' :
        return 'TSLA'
    elif symbol == 'NFLX' :
        return 'NFLX'
    elif symbol == 'BTC' :
        return 'BTC'
    elif symbol == 'ETH' :
        return 'ETH'
    elif symbol == 'BNB':
        return 'BNB'
    else : 
        return 'NONE'


df = get_data(symbol)
company = get_company_name(symbol)
st.header(company + '\tClose Price\n')
st.line_chart(df['Close'])
st.header(company + '\tVolume\n')
st.line_chart(df['Volume'])
st.header('\tStock Datas')
st.write(df.describe())


df=df[['Close']]
forecast=int(n)
df['Prediction']=df[['Close']].shift(-forecast)
x = np.array(df.drop(['Prediction'] , 1))
x = x[:-forecast]
y = np.array(df['Prediction'])
y=y[:-forecast]

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2)
mySVR = SVR(kernel='rbf' , C=1000 , gamma=0.1)
mySVR.fit(xtrain , ytrain)
SVMconf = mySVR.score(xtest , ytest)
st.header('SVM Accuracy')
st.success(SVMconf)

x_forecast=np.array(df.drop(['Prediction'] , 1))[-forecast :]
svmpred=mySVR.predict(x_forecast)
st.header('SVM Prediction')
st.success(svmpred)

lr = LinearRegression()
lr.fit(xtrain , ytrain)
lrconf = lr.score(xtest , ytest)
st.header('LR Accuracy')
st.success(lrconf)

lrpred = lr.predict(x_forecast)
st.header('LR Prediction')
st.success(lrpred)

#############################


df = get_data(symbol)
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[0:training_data_len , :]

xtrain = []
ytrain = []
n = 60

for i in range(n , len(training_data)) :
    xtrain.append(training_data[i-n:i , 0 ])
    ytrain.append(training_data[i , 0])

xtrain , ytrain = np.array(xtrain) , np.array(ytrain)
xtrain = np.reshape(xtrain , (xtrain.shape[0] , xtrain.shape[1] , 1))

model = Sequential()
model.add(LSTM(50 , return_sequences=True , input_shape=(xtrain.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error' , optimizer='adam')
model.fit(xtrain , ytrain , epochs=2 , batch_size=1)

test_data = scaled_data[training_data_len - n : , :]
xtest = []
ytest = dataset[training_data_len : , :]
for i in range(n , len(test_data)) :
    xtest.append(test_data[i-n : i , 0])

xtest = np.array(xtest)
xtest = np.reshape(xtest , (xtest.shape[0] , xtest.shape[1] , 1 ))

prediction = model.predict(xtest)
prediction = scaler.inverse_transform(prediction) 
rmse = np.sqrt(np.mean((prediction - ytest)**2))
st.header('RMSE : ')
st.success(rmse)

train = data[:training_data_len]  
valid = data[training_data_len:]
valid['prediction'] = prediction

# st.write('''
# # SYMBOL PREDICTOR ACCURACY

# ''')

# imag = Image.open("d:/IT/Pyton/accuracy.png")
# st.image(imag , width=600 )
# st.set_option('deprecation.showPyplotGlobalUse' , False)
# st.pyplot()

newdf = data[-60:].values
snewdf = scaler.transform(newdf)

xtest = []
xtest.append(snewdf)
xtest = np.array(xtest)
xtest = np.reshape(xtest , (xtest.shape[0] , xtest.shape[1] , 1))

pred = model.predict(xtest)
pred = scaler.inverse_transform(pred)
st.write('''
# DEEP LEARNING METHOD

''')
st.header('predicted price for next day ')
st.success(pred)

st.write('''
# SYMBOL PREDICTOR ACCURACY

''')

# imag = Image.open("d:/IT/Pyton/accuracy.png")
# st.image(imag , width=600 )
st.set_option('deprecation.showPyplotGlobalUse' , False)
st.pyplot()

plt.figure()
plt.title('PREDICTOR')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'prediction']])
plt.legend(['Train' , 'Value' , 'Prediction']) 
# plt.savefig('accuracy.png')