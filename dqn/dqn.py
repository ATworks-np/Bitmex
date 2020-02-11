import argparse
import models
import numpy as np

from collections import deque
from keras.optimizers import Adam


class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.input_shape = (12, 5)
        self.action_shape = (2,)

        self.priod = 1000 # bars


        self.ts = trade_system('Bitmex1h')
        self.found  = 10000 #dallrs

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
            for e in range(episodes):
                start_time = np.random.randint(0, len(self.ts.data) * 0.8 - (self.priod + self.input_shape[0]))
                for t in range(self.priod):
                    self.ts.time = start_time + t + self.input_shape[0]

                    action = self.choose_action()
                    reword, done = self.reword(action)

                    if done:
                        break


        # choosing action depending on epsilon
        def choose_action(self):
            if self.epsilon >= np.random.random():
                # randomly choosing action
                return np.random.randint(0,3)
            else:
                # choosing the best action from model.predict()
                return self.choose_best_action()

        def choose_best_action(self, state):
            inp1 = self.ts.data[self.ts.time - self.input_shape[0]:self.ts.time]
            if self.ts.position['dir'] == 'LONG':
                inp2 = [1,0,0]
            elif self.ts.position['dir'] == None:
                inp2 = [0,1,0]
            elif self.ts.position['dir'] == 'SHORT':
                inp2 = [0,0,1]
            action = self.model.predict([inp1,inp2])

            return

        def reword(self,action):
            if action == 0:
                ts.order_quantity('LONG',self.found)
            elif action == 2:
                ts.order_quantity('SHORT', self.found)
            prof = self.ts.profit + self.ts.calc_inprofit()
            if self.found - prof < self.found * 0.7: #loscut
                return -100, False
            else
                return  prof, True

class trade_system:
    def __init__(self,name):
        self.position = {'dir':None,'price':None,'volume':None}
        self.time = 0
        self.spread = 0.03
        self.profit = 0
        self.inprofit = 0

        self.data = np.loadtxt('../data/%s.csv'%name, delimiter=',',skiprows=1,usecols=[2,3,4,5,6])
        self.time = 0

    def updata_time(self):
        self.time += 1

    def get_price(self,dir):
        res = self.data[self.time,3]
        if dir == 'LONG':
            res *= (1 + self.spread/100)
        elif dir == 'SHORT':
            res *= (1 - self.spread/100)
        return res

    def order_quantity(self,dir = None,quantity = 0):
        if dir != None:
            price = self.get_price(dir)
            volume = quantity/price
            self.upgrade_poition(dir, price, volume)

    def order(self,dir = None,volume = 0):
        if dir != None:
            price = self.get_price(dir)
            self.upgrade_poition(dir, price, volume)


    def upgrade_poition(self, dir, price, volume):
        if self.position['dir'] == None:
            self.position['dir'] = dir
            self.position['price'] = price
            self.position['volume'] = volume
        elif self.position['dir'] == dir:
            self.position['price'] = (self.position['price'] * self.position['volume'] + price * volume) / (self.position['volume'] + volume)
            self.position['volume'] += volume
        elif self.position['dir'] != dir:
            if self.position['volume'] > volume:
                self.update_profit(price, volume)
                self.position['volume'] -= volume
            else:
                self.update_profit(price, self.position['volume'])
                self.position['dir'] = dir
                self.position['price'] = price
                self.position['volume'] = volume - self.position['volume']

        if self.position['volume'] == 0:
            self.position['dir'] = None
            self.position['price'] = None
            self.position['volume'] = None

    def update_profit(self, price, volume):
        if self.position['dir'] == 'LONG': # 買ポジの場合
            self.profit += (price - self.position['price']) * volume
        elif self.position['dir'] == 'SHORT':  # 売ポジの場合
            self.profit -= (price - self.position['price']) * volume

    def calc_inprofit(self):
        if self.position['dir'] == 'SHORT': # 売りポジの場合
            return -(self.get_price('LONG') - self.position['price']) * self.position['volume']
        elif self.position['dir'] == 'LONG':  # 買いポジの場合
            return  (self.get_price('SHORT') - self.position['price']) * self.position['volume']
        else:
            return 0



if __name__ == '__main__':
    ts = trade_system('Bitmex1h')
    print()
