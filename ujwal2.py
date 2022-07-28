import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.selectbox ('Enter Stock Ticker', ('AAPL' , 'TSLA' , 'PYPL' , 'V' ,'ENPH', '^NSEI','TTM','SBUX','SBIN.NS','MGAM','BTC-USD', 'GC=F', 'NFT-USD', 'ETH-USD'))
df = data.DataReader(user_input, 'yahoo', start,end)

#Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#Visualizations
st.subheader('Figure No.1 : Closing price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader(' Figure No.2 :Closing price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Figure No.3 :Closing price vs Time Chart with initiating after 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Figure No.4 :Closing price vs Time Chart with initiating after 400MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma400 = df.Close.rolling(400).mean()

fig = plt.figure(figsize = (12,6))
plt.plot(ma200, 'r')
plt.plot(ma400, 'g')
plt.plot(df.Close, 'y')
st.pyplot(fig)

st.subheader('Figure No.5 :Closing price vs Time Chart with initiating after 1000MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
ma400 = df.Close.rolling(400).mean()
ma1000 = df.Close.rolling(1000).mean()

fig = plt.figure(figsize = (12,6))
plt.plot(ma400, 'r')
plt.plot(ma1000, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


#Splitting data into X_train and Y_train

x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

model.save('keras_model.h5')

#LOAD MY MODEL
model = load_model('keras_model.h5')

#testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test= y_test * scale_factor

#Final Graph

st.subheader('Final Graph: Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',Label='Original Price')
plt.plot(y_predicted,'r',Label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
