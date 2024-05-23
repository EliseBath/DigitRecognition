import numpy as np
import time
image_size = 28 # width and length
image_pixels = image_size * image_size
data_path = "/mnt/c/Users/bathe/PycharmProjects/dif_rec_int/"
train_data = np.loadtxt("/mnt/c/Users/bathe/PycharmProjects/dif_rec_int/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("/mnt/c/Users/bathe/PycharmProjects/dif_rec_int/mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_img = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_img = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_lab = np.asfarray(train_data[:, :1])
test_lab = np.asfarray(test_data[:, :1])


lr = np.arange(10) #no_of_different_labels=10

# transform labels into one hot representation
train_labels_one_hot = (lr == train_lab).astype(np.float64)
test_labels_one_hot = (lr == test_lab).astype(np.float64)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


@np.vectorize
def ReLu(x):
    if(x > 0):              # bias aanpassen
        return x
    else:
        return 0


from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:
    transformfunctions = {
        'sigmoid': sigmoid,
        'ReLu': ReLu
    }


    def __init__(self, num_nodes, transformfunc, learning_rate):

        self.transformfunc = self.transformfunctions.get(transformfunc)
        self.number_nodes = num_nodes
        self.number_layers = len(num_nodes)
        self.no_of_in_nodes = num_nodes[0]          #weg
        self.no_of_out_nodes = num_nodes[2]          #weg
        self.no_of_hidden_nodes = num_nodes[1]          #weg
        self.learning_rate = learning_rate
        self.initialise()          #verandern naa initialise

        if transformfunc == "sigmoid":
            self.backpropagation = self.backpropagation_sigmoid
        elif transformfunc == "ReLu":
            self.backpropagation = self.backpropagation_ReLu

    def initialise(self):

        self.weights = []  # evt transformfunc als vecotr voor iedere laag

        for layer in range(self.number_layers - 1):  # -1 want gewichten zijn verbinding tussen lagen
            rad = 1 / np.sqrt(self.number_nodes[layer])
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            temp = X.rvs((self.number_nodes[layer + 1], self.number_nodes[layer]))
            self.weights.append(temp)

        self.bias = []
        for layer in range(1, self.number_layers):
            rad = 1 / np.sqrt(self.number_nodes[layer])
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            temp = X.rvs((self.number_nodes[layer],1))
            self.bias.append(temp)

        self.nodes_A = []
        for layer in range(self.number_layers):
            temp = np.zeros(self.number_nodes[layer])
            self.nodes_A.append(temp)



    def feedforward(self, input):
        prev_layer = input
        self.nodes_A[0] = prev_layer
        for l in range(self.number_layers-1):
            current_layer = np.dot(self.weights[l],prev_layer) + self.bias[l]
            current_layer = self.transformfunc(current_layer)
            self.nodes_A[l+1] = current_layer
            prev_layer = current_layer
        return prev_layer


    def backpropagation_sigmoid(self, act_output, des_output):
        dC_dA = des_output - act_output  # dC_dA2*dA2_dZ^2
        # update the weights:
        for i in reversed(range(len(self.weights))):
            dC_dZ= dC_dA * self.nodes_A[i+1] * (1.0 - self.nodes_A[i+1])  # ipv sigma'
            self.bias[i] += self.learning_rate * dC_dZ
            dC_dW = np.outer(dC_dZ,self.nodes_A[i].T)  # *dZ2_dW2
            self.weights[i] += self.learning_rate *  dC_dW
            dC_dA = np.dot(self.weights[i].T, dC_dA)


    def backpropagation_ReLu(self, act_output, des_output):
        dC_dA = des_output - act_output  # dC_dA2*dA2_dZ^2
        # update the weights:
        for i in reversed(range(len(self.weights))):
            dA_dZ = (self.nodes_A[i+1] > 0).astype(int)
            dC_dZ = dC_dA * dA_dZ # ipv sigma'
            self.bias[i] += self.learning_rate * dC_dZ
            dC_dW = np.outer(dC_dZ, self.nodes_A[i].T)  # *dZ2_dW2
            self.weights[i] += self.learning_rate * dC_dW
            dC_dA = np.dot(self.weights[i].T, dC_dA)

    def learn(self, input, des_output):
        input = np.array(input, ndmin=2).T
        des_output = np.array(des_output, ndmin=2).T
        tic = time.perf_counter()
        output = self.feedforward(input)
        toc = time.perf_counter()
        print(f"feedforw: {toc - tic:0.4f} seconds\n")


        tic = time.perf_counter()
        self.backpropagation(output, des_output)
        toc = time.perf_counter()
        print(f"backprop: {toc - tic:0.4f} seconds\n")


    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.feedforward(np.array(data[i], ndmin=2).T)
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


ANN = NeuralNetwork(num_nodes=[image_pixels, 100, 10], transformfunc="ReLu", learning_rate=0.1)

tic = time.perf_counter()
for i in range(len(train_img)):
    ANN.learn(train_img[i], train_labels_one_hot[i])
toc = time.perf_counter()
print(f"learn: {toc - tic:0.4f} seconds\n")

for i in range(20):
    res = ANN.feedforward(np.array(test_img[i], ndmin=2).T)
    print(test_lab[i], np.argmax(res), np.max(res))

corrects, wrongs = ANN.evaluate(train_img, train_lab)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = ANN.evaluate(test_img, test_lab)
print("accuracy: test", corrects / ( corrects + wrongs))

