import time
import gzip
f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 50000

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data_image = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data_image = data_image.reshape((num_images, image_size*image_size))
# print(data_image)


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
    try:
        ans = exp(-x)
    except OverflowError:
        print('sig fout ' + x.astype(str) + '\n')
        ans = float('inf')
    return 1/(1+ans)

def sigmoid_der(x):
    try:
        ans = exp(-x)
    except OverflowError:
        print('sig fout ' + x.astype(str) + '\n')
        ans = float('inf')
    return ans/(1+ans)**2

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
        diff = actual_output - desired_output   #volgorde van termen is belangrijk en juist zo
        return 1/5*diff


    def feedforward(self, input):
        prev_layer = input
        self.nodes_A[0] = input
        for l in range(self.number_layers-1):
            current_layer = self.weights[l].dot(prev_layer) + self.bias[l]
            # print(l)
            # for node in range(len(current_layer)):
                # current_layer[node] = self.transformfunc(current_layer[node])
            self.nodes_Z[l+1] = current_layer
            current_layer = [self.transformfunc(i) for i in current_layer]
            self.nodes_A[l+1] = current_layer
            prev_layer = current_layer
        return prev_layer


    def backpropagation(self, output, des_output):
        learning_rate = 0.01
        epsilon = 0.002
        dC_dA = self.cost_der(output, des_output)
        for i in reversed(range(len(self.weights))):
            dA_dZ = [sigmoid_der(l) for l in self.nodes_Z[i+1]]
            dC_dZ = dC_dA*dA_dZ


            dC_dA = np.dot(dC_dZ, self.weights[i])           #staat klaar voor volgende ronde, als ik hier niet aanpas zal met aangepaste weights worden berekend

            #pas gewichten aan
            dC_dW = np.outer(dC_dZ, self.nodes_A[i]) + epsilon*np.ones(np.shape(self.weights[i]))
            # k = np.shape(self.weights[i])
            self.weights[i] = self.weights[i] - learning_rate*dC_dW         #past dit aan of immutable?

            #pas bias aan
            dC_dB = dC_dZ
            self.bias[i] = self.bias[i] - learning_rate*dC_dB
        return None

    def learn(self, input, des_number):
        #We berekenen eerst de waarden van de nodes in de outputlaag. De vector die deze waarden bevat noemen we 'act_output'.
        #De parameter 'des_number' wordt aan de functie meegegeven. Deze Als de i-de waarde van deze vector de grootste waarde is, denkt het neuraal netwerk het cijfer i te zien op de foto.
        #We zullen i 'max_number' noemen. We geven dit nummer samen met de gewenste output (het cijfer dat effectief te zien
        #is op de afbeelding) mee aan de functie 'backpropagation', deze zal de gewichten en bias geschikt aanpassen.
        act_output = self.feedforward(input)
        des_output = np.zeros(10)
        des_output[int(des_number)] = 1

        self.backpropagation(act_output, des_output)
        return None


    def __init__(self, num_nodes, transformfunc):      #nodes bevat alleen aantal hidden
        #number_nodes is een array die het aantal neuronen per laag bevat. Aangezien we met
        # 8x28-afbeeldingen werken en de output-laag de cijfers 0-9 voorstellen,
        # zal het eerste cijfer in de array 784 zijn en de laatste 10. vb: [784, 13, 25, 10]
        self.number_nodes = num_nodes
        self.number_layers = len(num_nodes)


        self.weights = []                           #evt transformfunc als vecotr voor iedere laag


        for layer in range(self.number_layers-1):       #-1 want gewichten zijn verbinding tussen lagen
            temp = np.random.uniform(low=-1, high=1, size = (num_nodes[layer + 1], num_nodes[layer]))
            self.weights.append(temp)


        self.bias = []
        for layer in range(1,self.number_layers):
            temp = np.random.uniform(low=-1, high=1, size = num_nodes[layer])
            self.bias.append(temp)


        self.nodes_A = []
        for layer in range(self.number_layers):
            temp = np.zeros(num_nodes[layer])
            self.nodes_A.append(temp)



        self.nodes_Z = []
        for layer in range(self.number_layers):    #er wordt ook Z voor input laag aangemaakt maar blijft nul
            temp = np.zeros(num_nodes[layer])
            self.nodes_Z.append(temp)

        self.transformfunc = self.transformfunctions.get(transformfunc)
        self.transformfunc_der = self.transformfunctions.get(transformfunc+'_der')

        self.activation = sigmoid   #fout?


N = NeuralNetwork([784,15,10], "sigmoid")
print(N.weights)
print('\n')
print('gewichtjes')
print(N.bias)
print('\n')
print('bias')

# v = np.ones(784)
# v*=2
# l=np.array([5])
# N.learn(v,l)

tic = time.perf_counter()

for i in range(len(data_labels)):
    N.learn(data_image[i], data_labels[i])


print(N.weights)
print('\n')
print('gewichtjes2')
print(N.bias)
print('\n')
print('bias')



toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

# N = NeuralNetwork([2,4,10], "sigmoid")
# v=np.array([1,3])
# N.learn(v,2)




v=np.array([1,3])
# n=np.array([2,1])         #maakt niet uit of vecotr getransponeerd of niet, plaats in haakjes maakt uit
# temp = np.array([[1,1,0],[0,2,3]])          #matrix transponeren geeft verschil
# # print(np.outer(n,v))                         #transponeert matrix niet automatisch als verkeerde dim
# # print(n.transpose())
# k=np.dot(v.transpose(),temp)
# print(np.outer(n,v))

# v = [sigmoid_der((i)) for i in v]
# print(np.ones(np.size(v)))
# print(np.size(v))





