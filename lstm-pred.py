import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

company_ticker = "AAPL" # Ticker der Apple Aktie
train_von = "2010-12-16" # Startpunkt der Trainingsdaten
train_bis = "2021-03-01" # Endpunkt der Trainingsdaten
test_bis = datetime.now() # Startpunkt der Testdaten zur Validierung

# Herunterladen der Trainingsdaten
datensatz = yf.download(company_ticker , start=train_von, end=train_bis)
# Herunterladen der Testdaten
datensatz_test = yf.download(company_ticker , start=train_bis, end=test_bis)

train_aktien_preise = datensatz['Close'].values
test_aktien_preise = datensatz_test['Close'].values 

from sklearn.preprocessing import MinMaxScaler

# Festlegen der Min & Max-Werte der Normalisierung 
sc = MinMaxScaler(feature_range = (0, 1)) 
# Anwendung der Skalierung auf den Trainingsdatensatz 
training_datensatz_skaliert = sc.fit_transform(train_aktien_preise.reshape(-1,1)) 


import numpy as np
X_train = []
y_train = []

# Anzahl der Schlusskurse, die in das Modell kommen 
zeitraum_der_vorhersage = 10
# Transformieren der Daten in einzelne Sequenzen 
# Eine Sequenz beinhaltet dann 10 Schlusskurse 
for i in range(zeitraum_der_vorhersage, len(training_datensatz_skaliert)):
    X_train.append(training_datensatz_skaliert[i-zeitraum_der_vorhersage:i, 0])
    y_train.append(training_datensatz_skaliert[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True
))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mean_squared_error", "mean_absolute_error"])

regressor.fit(X_train, y_train, epochs = 100, batch_size = 100)




datensatzset_total = pd.concat((datensatz['Close'], datensatz_test['Close']), axis = 0)

inputs = datensatzset_total[len(datensatzset_total) - len(test_aktien_preise) - zeitraum_der_vorhersage:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(zeitraum_der_vorhersage, len(inputs)):
    X_test.append(inputs[i-zeitraum_der_vorhersage:i, 0])

X_test = np.array(X_test)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_prices = regressor.predict(X_test)
predicted_prices = sc.inverse_transform(predicted_prices)


plt.plot(test_aktien_preise, color="black", label=f"Actual {company_ticker}  Price")
plt.plot(predicted_prices, color="red", label=f"Predicted {company_ticker}  Price")
plt.title(f"{company_ticker}  Stock Price Predicition")
plt.show()
test = list(train_aktien_preise) + list(test_aktien_preise)
plt.plot(test, color="green", label=f"Actual Apple Price")
plt.show()

real_datensatz = [inputs[len(inputs)+1 - zeitraum_der_vorhersage:len(inputs+1), 0]]
real_datensatz = np.array(real_datensatz)
print(real_datensatz.shape)
real_datensatz = np.reshape(real_datensatz, (real_datensatz.shape[0], real_datensatz.shape[1], 1))
print(real_datensatz.shape)

prediction =  regressor.predict(real_datensatz)
prediction = sc.inverse_transform(prediction)
print(prediction)