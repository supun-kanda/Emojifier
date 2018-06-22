import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

word2Vec = {}

#Extracting word2vec from textfile given
with open('glove.6B.50d.txt', 'r', encoding="utf8") as file:
    for line in file:
        line = line.strip().split()
        current_word = line[0]
        word2Vec[current_word] = np.array(line[1:], dtype = np.float64)

#reading csv input training & testing files
def read_file(file):
    preSet = np.zeros((1,2))
    with open(file, 'r') as file:
        csvFile = csv.reader(file)
        for row in csvFile:
            preSet = np.vstack((preSet, row[:2]))
    preSet = preSet[1:,:]
    return preSet

preTrainSet = read_file('train_emoji.csv')
preTestSet = read_file('tesss.csv')

time_step  = len(max(preTrainSet[:,0], key=len).split())
n = 50

def dataSetForYou(preSet):
    m = preSet.shape[0]
    X = np.zeros((m,n,time_step))
    Y = np.zeros((m,5))
    for i in range(m):
        words = preSet[i,0].lower().split()
        vec = [word2Vec[word] for word in words]
        Y[i, int(preSet[i,1])] = 1
        for j in range(len(vec)):
            X[i, :, j] = np.array(vec[j])
    X = X.reshape(m, time_step, n)
    return X, Y, m

TrainX, TrainY, m = dataSetForYou(preTrainSet)
TestX, TestY, m_test = dataSetForYou(preTestSet)

print('DataSet Created')
print("examples:Train-%d Test-%d, dimensions:%d, time steps:%d" %(m, m_test, n, time_step))

model = Sequential()
model.add(LSTM(20, input_shape=(time_step,n)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(TrainX, TrainY, epochs=50, batch_size=1, verbose=2)

score = model.evaluate(TestX, TestY, batch_size=1, verbose=0)

'''
output = model.predict(TestX)

predict = np.argmax(output, axis=1)
actual = np.argmax(TestY, axis=1)

for i in range(len(actual)):
    print("%d - %d" %(actual[i],predict[i]))
'''

def emojify(input):
    x_in = np.zeros((1, n, time_step))
    compressed = input.lower().split()
    vec = [word2Vec[word] for word in compressed]
    for j in range(time_step):
            x_in[0, :, j] = np.array(vec[j])
            if(len(vec)<time_step):
                break
    x_in = x_in.reshape(1, time_step, n)
    predict = model.predict(x_in)
    predict = np.argmax(predict, axis=1)
    if(predict==0):
        print("Lovely")
    elif(predict==1):
        print("Sporty")
    elif(predict==2):
        print("Haha")
    elif(predict==3):
        print("Oh That's sad")
    elif(predict==4):
        print("I heard Food")

s=""
while(True):
    if(s=="q"):
        break
    else:
        s=input("Input: ")
        emojify(s)

