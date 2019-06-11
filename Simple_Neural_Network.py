import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#InputData = np.array([[3,   1.5],
#                    [2,   1],
#                    [4,   1.5],
#                    [3,   1],
#                    [3.5, .5],
#                    [2,   .5],
#                    [5.5,  1],
#                    [1,    1]])

#TargetData = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])

#mystery_flower = np.array([[4.5, 1]])

InputData = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])

TargetData = np.array([[0], [0], [0], [1], [1], [1]])

TestData = np.array([[0, 1, 1],
                     [1, 0, 0]])

w1 = np.zeros((3, 1))
b1 = np.random.random()

iterations = 10000
lr = 0.1
costlist = []

for i in range(iterations):

    random = np.random.choice(len(InputData))

    if i % 100 == 0:
        c = 0
        for j in range(len(InputData)):
            ze1 = np.dot(w1.T, InputData[j].reshape(3, 1)) + b1
            ae1 = sigmoid(ze1)
            c += float(np.square(ae1 - TargetData[j]))
        costlist.append(c)

    z1 = np.dot(w1.T, InputData[random].reshape(3, 1)) + b1
    a1 = sigmoid(z1)

    cost = np.square(a1 - TargetData[random])

    dcost = 2 * (a1 - TargetData[random])
    dz1 = sigmoid_p(a1)
    dw1 = InputData[random].reshape(3, 1)

    w1 = w1 - lr * dcost * dz1 * dw1

    b1 = b1 - lr * dcost * dz1

for x in range(len(InputData)):

    z1 = np.dot(w1.T, InputData[x].reshape(3, 1)) + b1
    a1 = sigmoid(z1)

    cost = np.square(a1 - TargetData[x])
    print(InputData[x])
    print("prediction: ", a1)
    print("cost: ", cost, "\n")

plt.plot(costlist)
plt.show()
