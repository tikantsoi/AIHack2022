# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
from scipy import stats
import os
import yfinance as yf

def import_data(codes, start_date, end_date):
    """downloads all data for each backtest from yahoo Finance."""
    data = yf.download(codes, start_date, end_date)
    # if only one stock code is entered data is reformated so that
    # it is the same format as when multiple stocks are entered
    if len(codes) == 1:
      data.columns = [data.columns, codes*len(data.columns)]
    #returns data where any row containing NAN values is removed
    return data.dropna()

def load_data(file):
    dataframe = pd.read_csv(file, index_col=0)
    dataframe = dataframe.drop(columns=['Time', 'Net Imbalance Value'])
    dataframe.fillna(method="ffill", inplace=True) # if nan if value above
    dataset = dataframe.values
    
    return dataset

# look_forward is defined from the last data point in the kernel
def preprocess(dataset, kernel_width, look_forward): 
    
    split_ratio = 0.2 # ratio split into test
    train_size = int(len(dataset) * (1-split_ratio))
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    
    scaler = MinMaxScaler(feature_range=(0, 1)) # for sigmoid
    train = scaler.fit_transform(train)
    test  = scaler.transform(test)
    
    def dataSegment(split_dataset):
        X = list()
        y = list()
        
        for i in range(len(split_dataset) - look_forward - kernel_width + 1):
            X.append(split_dataset[i : i + kernel_width])
            y.append(split_dataset[i + kernel_width + look_forward - 1][0]) # hard coded to only store the 0th dimension EPEX
            
        return np.array(X), np.array(y)
   
    Xtrain, ytrain = dataSegment(train)
    Xtest, ytest = dataSegment(test)
  
    # data now in [samples, features]
    # reshape input to be [samples, time steps, features]
    Xtrain = np.reshape(Xtrain, (len(Xtrain), Xtrain.shape[1], Xtrain.shape[2]))
    Xtest = np.reshape(Xtest, (len(Xtest), Xtrain.shape[1], Xtest.shape[2]))
    
    print(f"Xtrain shape: {Xtrain.shape}")
    print(f"Xtest shape: {Xtest.shape}")

    print(f"ytrain shape: {ytrain.shape}")
    print(f"ytest shape: {ytest.shape}")
    
    return Xtrain, ytrain, Xtest, ytest, scaler
            
def train_model(Xtrain, ytrain, Xtest, ytest, epochs, learning_rate, batch_size, store_model = False, model_name = ""):
    model = keras.Sequential(
        [
            layers.LSTM(64, input_shape=(Xtrain.shape[1], Xtrain.shape[2])),
            layers.Dropout(0.4),
            layers.Dense(1)
        ]
    )
    model.summary()
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=["mean_absolute_percentage_error"])
    
    history = model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    score = model.evaluate(Xtest, ytest)
    print("Test loss (mean_squared_error) :", score[0])
    print("Test metrics (mean_absolute_percentage_error) :", score[1])
    
    if store_model:
        if not os.path.exists("Models"):
            os.makedirs("Models")
        model.save(model_name+".h5")
        with open(f"{model_name}.obj","wb") as f0:
            pickle.dump(history.history, f0)
        
    return model, history, score

def plot_loss(history):
    Epoch = np.arange(1, len(history['loss']) + 1, 1)
    plt.plot(Epoch, history['loss'], label='loss')
    plt.plot(Epoch, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Mean Squared Error]')
    plt.legend()
    plt.grid(True)

def predict(model, dataset, raw_data=False):
    predictions = model.predict(dataset) # Xtest if not raw data
    if raw_data: # raw data requires the same processing as above
        data = scaler.transform(dataset)
        X = list() # require the same processing as before to group data into kernel
        for i in range(len(data) - look_forward - kernel_width + 1):
            X.append(data[i : i + kernel_width])
        X = np.array(X)
        X = np.reshape(X, (len(X), X.shape[1], X.shape[2]))
        predictions = model.predict(X) # predict on X after processing

    predictions_copy = np.repeat(predictions,scaler.n_features_in_, axis=-1)
    predictions = scaler.inverse_transform(predictions_copy)
    predictions = predictions.T[0]
    
    return predictions

def evaluate(predictions, answers, scaler, model_name, raw_data=False):
    x = np.linspace(predictions.min(), predictions.max(),1000)

    if raw_data == False:
        answers = np.expand_dims(answers, -1)
        answers_copy = np.repeat(answers,scaler.n_features_in_, axis=-1)
        answers = scaler.inverse_transform(answers_copy)
        answers = answers.T[0] # change
        
    plt.scatter(answers, predictions, label="Data")
    plt.plot(x,x, '--', color='black', label="Fit")
    plt.xlabel("True value", fontsize=14)
    plt.ylabel("Prediction", fontsize=14)
    plt.legend()
    rsquared = stats.linregress(predictions.reshape(len(predictions)), answers.reshape(len(answers)))
    plt.title(f"R squared: {rsquared[2]**2}")
    plt.show()
    plt.plot(answers)
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.show()
    if not os.path.exists("Plots"):
        os.makedirs("Plots")
    plt.savefig(f"Plots/{model_name}.png")
        




