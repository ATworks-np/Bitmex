import argparse
import models
import numpy as np

from collections import deque
from keras.optimizers import Adam


class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.input_shape = (12, 1)
        self.action_shape = (2,)

        self.volume  = 1

        self.action = [0, 1, 0]
        self.position_value = None

        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0     # randomness of choosing random action or the best one
        self.e_decay = 0.9999  # epsilon decay rate
        self.e_min = 0.01      # minimum rate of epsilon

        self.learning_rate = 0.0001
        self.opt= Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        self.my_model = models.load_lstm(self.input_shape,self.action_shape)
        self.my_model.compile(loss="mse", optimizer=self.opt)

        def remember_memory(self, state, action, reward, next_state, next_movables, done):
            self.memory.append((state, action, reward, next_state, next_movables, done))

        def train(self):
            episodes = 100
            ts = 100
            for e in range(episodes):

                for t in range(ts):
                    self.action = self.choose_action()
                    reword, done = self.reword()

                    if done:
                        break


        # choosing action depending on epsilon
        def choose_action(self, state):
            if self.epsilon >= np.random.random():
                # randomly choosing action
                n = np.random.random()
                return [n, 1-n]
            else:
                # choosing the best action from model.predict()
                return self.choose_best_action(state)

        def choose_best_action(self, state):
            inp = 0
            self.action = self.model.predict([inp,self.action])

            return

        def reword(self):

class trade_system:
    def __init__(self):
        self.position = {'statas':None,'price':None}
        self.time = 0
    def get_price(self):
        return

    def buy(self,volume):
        value =  self.get_price()
        price = value * volume
        if self.position['statas'] == None:
            self.position = {'statas': 'LONG', 'price': price}
            return 0
        elif self.position['statas']  == 'LONG':
            self.position['price'] += price
            return
        elif self.position['statas']  == 'SHORT':
            res = self.position['price'] - price
            self.position = {'statas': None, 'price': None}
            return res

    def sell(self,volume):
        value =  self.get_price()
        price = value * volume
        if self.position['statas'] == None:
            self.position = {'statas': 'SHORT', 'price': price}
            return 0
        elif self.position['statas']  == 'LONG':
            res = price - self.position['price']
            self.position = {'statas': None, 'price': None}
            return res
        elif self.position['statas']  == 'SHORT':
            self.position['price'] += price
            return 0







if __name__ == '__main__':

