import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout, Activation
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime

#Ensuring the use of gpu tensorflow

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

#Enter info regarding model

look_back = 60
forward_days = 1
num_periods = 20

#data

today = datetime.today().strftime('%Y-%m-%d')
df = yf.download("BTC-USD", start="2014-01-01", end=today)
df = df["Close"]

array = df.values.reshape(df.shape[0],1)
scl = MinMaxScaler()
array = scl.fit_transform(array)

division = len(array) - num_periods*forward_days
array_test = array[division-look_back:]
array_train = array[:division]

def processData(data, look_back, forward_days,jump=1):
    X,Y = [],[]
    for i in range(0,len(data) -look_back -forward_days +1, jump):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+forward_days)])
    return np.array(X),np.array(Y)

X_test,Y_test = processData(array_test,look_back,forward_days,forward_days)
Y_test = np.array([list(a.ravel()) for a in Y_test])

X,y = processData(array_train,look_back,forward_days)
y = np.array([list(a.ravel()) for a in y])

X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=0.20, random_state=42)

#Training the model

class MyHyperModel(HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        
        # Specify model
        model = Sequential()
        NUM_NEURONS_FirstLayer = hp.Choice('NUM_NEURONS_FirstLayer',values=[80, 90, 100, 120, 140])

        model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(look_back,1), return_sequences=True))

        model.add(Dropout(hp.Float('dropout_1', 0, 0.4, step=0.1, default=0)))

        
        NUM_NEURONS_SecondLayer = hp.Choice('NUM_NEURONS_SecondLayer',values=[30, 40, 50, 60, 70])
        model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))

        model.add(Dropout(hp.Float('dropout_2', 0, 0.4, step=0.1, default=0)))

        # Output layer
        model.add(Dense(forward_days, activation='linear'))

        # Compile the constructed model and return it
        model.compile(
            optimizer=Adam(
                hp.Choice('learning_rate',
                          values=[0.0005, 0.001, 0.0015, 0.002])),
            loss='mean_squared_error')

        return model
    


hypermodel = MyHyperModel(num_classes=1)    

# Construct the RandomSearch tuner
random_tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=100,
    executions_per_trial = 1,
    seed=10, 
    project_name='lstm-kerastuner-uni',
    directory="C:\\PATH")

# Search for the best parameters of the neural network using the contructed random search tuner
random_tuner.search(X_train, Y_train,
             epochs=100,
             validation_data=(X_validate, Y_validate))

#get the best model

random_params = random_tuner.get_best_hyperparameters()[0]
best_model = random_tuner.get_best_models(1)[0]

#Evaluate it on the validation test

print("Evalutation of best performing model:")
print(best_model.evaluate(X_validate, Y_validate))

#Get summary

#random_tuner.results_summary()

#Saving the model

file_name = 'LSTM_BTC_tuned.h5'
best_model.save(file_name)
print("Saved model `{}` to disk".format(file_name))

#Predict 

Xt = best_model.predict(X_test)

plt.figure(figsize = (15,10))

        
plt.plot(scl.inverse_transform(Xt.reshape(-1,1)), color='r', label='Prediction')  
plt.plot(scl.inverse_transform(Y_test.reshape(-1,1)), label='Target')
plt.xticks(np.arange(0, 20, 1), np.arange(1, 21, 1))
plt.xlabel('Days')
plt.ylabel('USD')
plt.legend(loc='best')
plt.show()