# Pre-processing 
import math, sys
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

#region Functions
def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []

    for i in range(I):
        m.append([fill]*J)

    return m


def ChangeDataFormat(dataset, formerDataLen):
    ''' Change dataset's form to fit in our RNN data structure
    Parameters
    -----------
    dataset : The group of data which we want to use for predict the results

    formerDataLen : The length of the sequence datas for predict the next data
    '''
    formerData = []
    predictData = []
    for i in range(len(dataset)-formerDataLen):
        data = dataset[i:(i+formerDataLen), 0]
        formerData.append(data)
        predictData.append(dataset[i+formerDataLen, 0])
    return np.array(formerData), np.array(predictData)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(y):
    return sigmoid(y)*(1-sigmoid(y))

def ReLU(x):
    return max(0.01*x, x)

def dReLU(y):
    if y<0:
        return 0.01
    elif y>=0:
        return 1
#endregion
#region DenseLayer
class Dense:

    def __init__(self, ni, nh, no):
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)

        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = ReLU(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        output_deltas = [0.0] * self.no

        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh

        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k] 
            hidden_deltas[j] = dReLU(self.ah[j]) * error

        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]

                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change 

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]

                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]

                self.ci[i][j] = change

        error = 0.0

        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2

        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        
        for i in range(self.ni):
            print(self.wi[i])

        print()
        print('Output weights:')

        for j in range(self.nh):
            print(self.wo[j])

    def train(self, x, y, iterations=1000, N=0.5, M=0.1):
        for i in range(iterations):
            error = 0.0
            for xi, yt in zip(x, y):
                inputs = xi
                targets = [yt]

                self.update(inputs)

                error = error + self.backPropagate(targets, N, M)

            if i % 100 == 0:
                print('error %-.5f' % error)

#endregion
#region RNN
class RNN:
    ''' Recurrent Neural Network '''
    def __init__(self, cells_count=10, batch_size=32, learning_rate=0.1):
        # Counts of cells in recursive layer
        self.cells_count = cells_count
        self.batch_size=batch_size
        self.lr = learning_rate
        self.w1h = None
        self.w1x = None
        self.b1 = None
        # Fully connected nn
        self.w2 = None
        self.b2 = None
        self.cells = None
        # Calculate error in training
        self.losses = []
        # Calcultae error in test
        self.val_losses = []
    
    def forpass(self, x):
        ''' Forpass calculating output '''
        # Initialize cells
        self.h = [np.zeros((x.shape[0], self.cells_count))]
        
        seq = np.swapaxes(x, 0, 1)
        # x.shape = (3, 1, 3) -> (batch_size, 1, formerDataLen)
        # swap -> (1, 3, 3) -> 1 group has 3 batch data, 1 batch data contains 3 data

        for x in seq:
            z1 = np.dot(x, self.w1x) + np.dot(self.h[-1], self.w1h) + self.b1
            h = np.tanh(z1)
            self.h.append(h)
            z2 = np.dot(h, self.w2) + self.b2
        return z2
    
    def backprop(self, x, err):
        ''' Backward propagation for re-grading weights '''
        m = len(x)

        w2_grad = np.dot(self.h[-1].T, err) / m
        b2_grad = np.sum(err) / m

        seq = np.swapaxes(x, 0, 1)

        w1h_grad = w1x_grad = b1_grad = 0

        err_to_cell = np.dot(err, self.w2.T) * (1-self.h[-1] **2)

        for x, h in zip(seq[::-1][:3], self.h[:-1][::-1][:1]):
            w1h_grad += np.dot(h.T, err_to_cell)
            w1x_grad += np.dot(x.T, err_to_cell)
            b1_grad += np.sum(err_to_cell, axis =0)

            err_to_cell = np.dot(err_to_cell, self.w1h) * (1 - h ** 2)

        w1h_grad /= m
        w1x_grad /= m
        b1_grad /= m

        return w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad

    def sigmoid(self, z):
        ''' Sigmoid activation function '''
        a = 1 / (1 + np.exp(-z))
        return a
    
    def init_weights(self, n_features, n_classes):
        ''' Init weights '''
        orth_init = tf.initializers.Orthogonal()
        glorot_init = tf.initializers.GlorotUniform()

        self.w1h = orth_init((self.cells_count, self.cells_count)).numpy()
        self.w1x = glorot_init((n_features, self.cells_count)).numpy()
        self.b1 = np.zeros(self.cells_count)
        self.w2 = glorot_init((self.cells_count, n_classes)).numpy()
        self.b2 = np.zeros(n_classes)

    def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
        y = y.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        # Reset numpy random seed
        np.random.seed(42)
        self.init_weights(x.shape[2], y.shape[1])

        for i in range(epochs):
            print('에포크', i, end=' ')
            batch_losses = []
            for x_batch ,y_batch in self.gen_batch(x, y):
                print('.', end=' ')
                a = self.training(x_batch, y_batch)
                a = np.clip(a, 1e-10, 1-1e-10)
                loss = np.mean(-(y_batch*np.log(a) + (1-y_batch) * np.log(1-a)))
                batch_losses.append(loss)
            print()
            self.losses.append(np.mean(batch_losses))
            self.update_val_loss(x_val, y_val)

    def gen_batch(self, x, y):
       length = len(x)
       bins = length // self.batch_size
       if length % self.batch_size:
           bins += 1
    #    indexes = np.random.permutation(np.arange(len(x)))
    #    x = x[indexes]
    #    y = y[indexes]
       for i in range(bins):
           start = self.batch_size * i
           end = self.batch_size * (i + 1)
           yield x[start:end], y[start:end]

    def training(self, x, y):
        z = self.forpass(x)
        a = self.sigmoid(z)
        err = -(y - a)
        w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
        # Update weights in cells
        self.w1h -= self.lr * w1h_grad
        self.w1x -= self.lr * w1x_grad
        self.b1 -= self.lr * b1_grad
        # Update weights in fully connected layer
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a

    def predict(self, x):
        z = self.forpass(x)
        return z
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1, 1))

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.sigmoid(z)
        a = np.clip(a, 1e-10, 1-1e-10)
        val_loss = np.mean(-(y_val*np.log(a) + (1-y_val) * np.log(1-a)))
        self.val_losses.append(val_loss)
#endregion



# Read csv files
df = read_csv('corona_daily.csv', usecols=[3], engine='python', skipfooter=3)
dataSet = df.values
dataSet = dataSet.astype('float32')

# Change dataset values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
Dataset = scaler.fit_transform(dataSet)

# Split dataset to two groups (train group, test group)
train_data, test_data = train_test_split(Dataset, test_size=0.2, shuffle=False)

# Change dataset's format for our RNN structure
# We will use former 3 datas for predicting 4th data
formerDataLen = 3
# Change train datas
x_train, y_train = ChangeDataFormat(train_data, formerDataLen)
# Change test datas
x_test, y_test = ChangeDataFormat(test_data, formerDataLen)

# Confirm the size of datasets
print("----------------------")
print("Size of the train, test datasets")
print(f"> Train dataset size \nSource : {x_train.shape}, Prediction : {y_train.shape}")
print(f"> Test dataset size \nSource : {x_test.shape}, Prediction : {y_test.shape}")

# Group the source data by the length of former data length
X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# Confirm the arrays' shape
print("----------------------")
print("The shape of the train/test datasets")
print(f"> Train dataset : {X_train.shape}\n> Test dataset : {X_test.shape}")

# Create sequential model to solve the RNN with time step
NN_Model = RNN(cells_count=3, batch_size=3, learning_rate=0.01)
DenseLayer = Dense(3, 7, 1)

NN_Model.fit(X_train, y_train, epochs=100, x_val=X_test, y_val=y_test)
X_train2, Y_train2 = ChangeDataFormat(NN_Model.predict(X_train), formerDataLen)
X_test2, Y_test2 = ChangeDataFormat(NN_Model.predict(X_test), formerDataLen)
DenseLayer.train(X_train2, Y_train2)
DenseLayer.test(zip(X_test2, Y_test2))
 
plt.plot(DenseLayer.losses, label='train loss')
plt.plot(DenseLayer.val_losses, label='test loss')
plt.legend()
plt.show()