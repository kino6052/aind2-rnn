import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    s = np.asarray(series)

    for i in range(len(series) - window_size): # Sliding Window
        X.append(s[i:i+window_size])
        y.append(s[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(32,
               input_shape=(window_size, 1)))  # returns a sequence of vectors of dimension 32
    # now model.output_shape == (None, 32)
    # note: `None` is the batch dimension.

    # for subsequent layers, no need to specify the input size:

    # to stack recurrent layers, you must use return_sequences=True
    # on any recurrent layer that feeds into another recurrent layer.
    # note that you only need to specify the input size on the first layer.
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', '\xa0', '¢', '¨', '©', 'ã',  '-', '"', '$', '%', '&', "'", '(', ')', '*', '/', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    processed_text = ""
    for c in text:
        if c in punctuation:
            pass
        else:
            processed_text += c
    return processed_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    s = text

    for i in range(int((len(text) - window_size)/step_size)): # Sliding Window
        try:
            i_step = i * step_size
            inputs.append(s[i_step:i_step+window_size])
            outputs.append(s[i_step+window_size])
        except Exception as e:
            pass
    # reshape each 
    # inputs = np.asarray(inputs)
    # inputs.shape = (np.shape(inputs)[0:2])
    # outputs = np.asarray(outputs)
    # outputs.shape = (len(outputs),1)

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200,
               input_shape=(window_size, num_chars)))  # returns a sequence of vectors of dimension 32
    # now model.output_shape == (None, 32)
    # note: `None` is the batch dimension.

    # for subsequent layers, no need to specify the input size:

    # to stack recurrent layers, you must use return_sequences=True
    # on any recurrent layer that feeds into another recurrent layer.
    # note that you only need to specify the input size on the first layer.
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    return model
