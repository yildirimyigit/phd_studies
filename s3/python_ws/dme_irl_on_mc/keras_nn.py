"""
  @author: yigit.yildirim@boun.edu.tr
"""

import keras
from keras.models import Sequential
from keras.layers import Dense


class KNN:
    def __init__(self):
        model = Sequential()
        model.add(Dense(64, kernel_initializer='uniform', activation='sigmoid', input_dim=2))
        model.add(Dense(256, kernel_initializer='uniform', activation='sigmoid'))
        model.add(Dense(256, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='tanh'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
