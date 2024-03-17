import gzip
f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data_image = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data_image = data_image.reshape((num_images, image_size*image_size))
print(data_image)


# import matplotlib.pyplot as plt
# image = np.asarray(data[4]).squeeze()
# plt.imshow(image)
# plt.show()

data_labels = np.empty(num_images)
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
for i in range(num_images):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    data_labels[i] = labels[0]

# from mlxtend.data import loadlocal_mnist
# import platform
#
# X, y = loadlocal_mnist(
#             images_path='train-images.idx3-ubyte',
#             labels_path='train-labels.idx1-ubyte')




from math import exp

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_der(x):
    return exp(-x)/(1+exp(-x))**2

def threshold(x):
    if(x < 1):              # bias aanpassen
        return 0
    else:
        return 1

class NeuralNetwork:

    transformfunctions = {
        'sigmoid': sigmoid,
        'sigmoid_der': sigmoid_der,
        'threshold': threshold
    }

    def cost(self, actual_output,desired_output):
        diff = actual_output - desired_output
        return np.mean(diff)

    def cost_der(self, actual_output, desired_output):
        diff = actual_output - desired_output
        return 1/5*diff


    def feedforward(self, input):
        prev_layer = input
        self.nodes_A[0] = input
        for l in range(len(self.number_nodes)):
            current_layer = self.weights[l].dot(prev_layer) + self.bias[l]
            # print(l)
            # for node in range(len(current_layer)):
                # current_layer[node] = self.transformfunc(current_layer[node])
            current_layer = [self.transformfunc(i) for i in current_layer]
            prev_layer = current_layer
        current_layer = self.weights[len(self.weights)-1].dot(prev_layer) + self.bias[len(self.weights)-1]
        current_layer = [self.transformfunc(i) for i in current_layer]
        prev_layer = current_layer
        return prev_layer


    def backpropagation(self, output, des_output):
        learning_rate = 0.5
        dE_dA = self.cost_der(output, des_output)
        for i in reversed(range(len(self.weights))):
            dA_dZ = [sigmoid_der((l)) for l in self.nodes_Z[i]]
            dE_dZ = dE_dA*dA_dZ


            dE_dA = np.dot(dE_dZ, self.weights[i])           #staat klaar voor volgende ronde, als ik hier niet aanpas zal met aangepaste weights worden berekend

            #pas gewichten aan
            dE_dW = np.outer(dE_dZ, self.nodes_A[i])
            self.weights[i] = self.weights[i] - learning_rate*dE_dW

            #pas bias aan
            dE_dB = dE_dZ
            self.bias[i] = self.bias[i] - learning_rate*dE_dB
        return None

    def learn(self, input, des_output):
        act_output = self.feedforward(input)
        self.backpropagation(act_output, des_output)
        return None


    def __init__(self, nodes, transformfunc):       #nodes bevat alleen aantal hidden
        self.number_nodes = nodes


        self.weights = []                           #evt transformfunc als vecotr voor iedere laag
        temp = np.random.randint(1, 10, size=(nodes[0], 784))  # gewichtjes tussen eerste hidden layer en input
        self.weights.append(temp)

        for layer in range(len(nodes)-1):
            temp = np.random.randint(1, 10, size = (nodes[layer +1], nodes[layer]))
            self.weights.append(temp)
        temp = np.random.randint(1, 10, size=(10, nodes[len(nodes)-1]))        #gewichtjes tussen laatste hidden layer en output

        self.weights.append(temp)



        self.bias = []
        for layer in range(len(nodes)):
            temp = np.random.randint(1, 5, size = nodes[layer])
            self.bias.append(temp)

        temp = np.random.randint(1, 5, size=10)                  #bias voor laatste laag
        self.bias.append(temp)



        self.nodes_A = []
        self.nodes_A.append(np.zeros(784))
        for layer in range(len(nodes)):
            temp = np.zeros(nodes[layer])
            self.nodes_A.append(temp)
        self.nodes_A.append(np.zeros(10))


        self.nodes_Z = []
        for layer in range(len(nodes)):
            temp = np.zeros(nodes[layer])
            self.nodes_Z.append(temp)
        self.nodes_Z.append(np.zeros(10))

        self.transformfunc = self.transformfunctions.get(transformfunc)
        self.transformfunc_der = self.transformfunctions.get(transformfunc+'_der')

        self.act = sigmoid


N = NeuralNetwork([15], "sigmoid")
print(N.weights)
print(N.bias)
# v = np.ones(784)
# v*=2
# l=np.array([5])
# N.learn(v,l)
#
for i in range(5):
    N.learn(data_image[i], data_labels[i])
print(N.weights)
print(N.bias)







# v=np.array([1,2,1])
n=np.array([1,1,5])
temp = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3]])
# print(np.outer(n,v))
# print(n.transpose())
k=temp.dot(n)
print(temp.dot(n))

# v = [sigmoid_der((i)) for i in v]






