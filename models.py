from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
import tensorflow as tf
from tcn import TCN
import statsmodels.api as sm

def LSTM(time_step, features, x_train, y_train, x_val, y_val, unit_lstm_1, unit_lstm_2):
    model_lstm = Sequential()
    model_lstm.add(InputLayer((time_step, features)))
    model_lstm.add(
        LSTM(unit_lstm_1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model_lstm.add(LSTM(unit_lstm_2))
    model_lstm.add(Dense(8, 'relu'))
    model_lstm.add(Dense(1, 'linear'))
    model_lstm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model_lstm.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=120)
    return model_lstm, history

def CNN(time_step, features, x_train, y_train, x_val, y_val, unit_cnn):
    model_cnn = Sequential()
    model_cnn.add(InputLayer((time_step, features)))
    model_cnn.add(Conv1D(unit_cnn, kernel_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(1, 'linear'))
    # model_cnn.summary()
    model_cnn.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model_cnn.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=120)
    return model_cnn, history

def RNN(time_step, features, x_train, y_train, x_val, y_val, unit_rnn_1, unit_rnn_2):
    # RNN
    model_rnn = Sequential()
    model_rnn.add(InputLayer((time_step, features)))
    model_rnn.add(
        SimpleRNN(unit_rnn_1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model_rnn.add(SimpleRNN(unit_rnn_2))
    model_rnn.add(Dense(8, 'relu'))
    model_rnn.add(Dense(1, 'linear'))
    model_rnn.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model_rnn.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=120)
    return model_rnn, history


def GRU(time_step, features, x_train, y_train, x_val, y_val, unit_gru_1, unit_gru_2):
    # GRU
    model_gru = Sequential()
    model_gru.add(InputLayer((time_step, features)))
    model_gru.add(GRU(unit_gru_1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model_gru.add(GRU(unit_gru_2))
    model_gru.add(Dense(8, 'relu'))
    model_gru.add(Dense(1, 'linear'))
    # model_gru.summary()
    model_gru.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model_gru.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=120)
    return model_gru, history

def cnn_lstm(time_step, features, x_train, y_train, x_val, y_val, unit_cnn, unit_lstm_1, unit_lstm_2):
    model_cnn_lstm = Sequential()
    model_cnn_lstm.add(InputLayer((7, 9)))
    model_cnn_lstm.add(Conv1D(unit_cnn, kernel_size=2, return_sequences=True))
    model_cnn_lstm.add(
        LSTM(unit_lstm_1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model_cnn_lstm.add(LSTM(unit_lstm_2))
    model_cnn_lstm.add(Dense(8, 'relu'))
    model_cnn_lstm.add(Dense(1, 'linear'))

    model_cnn_lstm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model_cnn_lstm.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=120)
    return model_cnn_lstm, history

def tcn_BiGRU_attention(time_step, features, x_train, y_train, x_val, y_val):
    inputs = Input(batch_shape=(None, time_step, features))
    # TCN layer
    tcn = TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True)(inputs)
    # LSTM layer
    lstm = Bidirectional(GRU(50, return_sequences=True))(tcn)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(100)(attention)
    attention = Permute([2, 1])(attention)
    attention = Multiply()([lstm, attention])
    attention = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention)

    # Output layer
    outputs = Dense(1)(attention)

    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    history = model.fit(x_train, y_train, epochs=120, batch_size=32, validation_data=(x_val, y_val))
    return model, history

def ARIMA(train_x, train_y, test_x, test_y):
    model = sm.tsa.ARIMA(endog=train_y, exog=train_x, order=(1, 1, 1))
    fit_model = model.fit()
    forecast = fit_model.forecast(steps=len(test_y), exog=test_x)
    forecasted_values = forecast[0]
    return forecasted_values, test_y