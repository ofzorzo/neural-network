import argparse
import neural_network as nn
import numpy as np
import pandas as pd
from random import uniform

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def minmax_normalize(instances):
    scaler = MinMaxScaler()
    scaler.fit(instances)
    print(scaler.data_max_)
    return scaler.transform(instances)

def standard_normalize(instances):
    scaler = StandardScaler()
    scaler.fit(instances)
    return scaler.transform(instances)

def create_neural_network(network, weights, dataset):
    with open(network) as network:
        lambda_val = float(network.readline().split()[0])
        neurons = []
        for line in network:
            neurons.append(int(line))
        s = {
            'lambda' : lambda_val,
            'neurons' : neurons
        }

    with open(weights) as weights:
        w = []
        for line in weights:
            layer = []
            line = line.split(";")
            line[-1] = line[-1].strip()
            for i in range(len(line)):
                neuron = line[i].split(",")
                neuron = [float(j) for j in neuron]
                layer.append(neuron)
            #print(layer)
            np_layer = np.matrix(layer)
            w.append(np_layer)
        #print(w)
    
    with open(dataset) as dataset:
        train = []
        for instance in dataset:
            instance = instance.split(";")
            inputs = instance[0]
            inputs = inputs.strip()
            inputs = inputs.split(",")
            inputs = [float(j) for j in inputs]
            outputs = instance[1]
            outputs = outputs.strip()
            outputs = outputs.split(",")
            outputs = [float(j) for j in outputs]
            train.append([inputs, outputs])
        #print(train)
    return s, w, train

def print_network_parameters(s, w, train):
    print("Parametro de regularizacao lambda=" + "{:.3f}".format(s['lambda']))
    print("")
    print("Inicializando rede com a seguinte estrutura de neuronios por camada: [", end="")
    first = True
    for n in s['neurons']:
        if first:
            first = False
            print(n, end="")
        else:
            print(" ", end="")
            print(n, end="")
    print("]")
    print("")
    for k in range(0, len(w)):
        print("Theta" + str(k+1) + " inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
        for i in range(0, len(w[k])):
            print("\t\t", end="")
            for j in range(0, w[k][i].size):
                print("{:.5f}".format(w[k][i, j]), end=" ")
            print("")
        print("")
    print("")
    print("Conjunto de treinamento")
    exemplo = 1
    for instance in train:
        print("\tExemplo " + str(exemplo))
        print("\t\tx: [", end="")
        first = True
        for x in instance[0]:
            if first:
                print("{:.5f}".format(x), end="")
                first = False
            else:                
                print("   {:.5f}".format(x), end="")
        print("]")
        print("\t\ty: [", end="")
        first = True
        for y in instance[1]:            
            if first:
                print("{:.5f}".format(y), end="")
                first = False
            else:
                print("   {:.5f}".format(y), end="")
        print("]")
        exemplo += 1
    print("")
    print("--------------------------------------------")

#assume que só tem um valor a ser previsto
def create_train_set(dataset, predicted_index, drop_col, drop_row):
    data = pd.read_csv(dataset, delimiter=',', header=None)
    if drop_col != None:
        data = data.drop(columns=[drop_col])
    if drop_row != None:
        data = data.drop([drop_row])
    x = data.drop(columns=[predicted_index])
    y = data.drop(columns=[ c for c in range(len(data.columns)) if c!=predicted_index ])
    train = []
    instances_atributes = []
    for i in range(len(data)):
        instances_atributes.append(x.iloc[i, : ].tolist())

    if standard_normalize:
        normalized_instances = standard_normalize(instances_atributes)
    else:
        normalized_instances = minmax_normalize(instances_atributes)

    for i in range(len(data)):
        instance = []
        instance.append(normalized_instances[i].tolist())
        instance.append(y.iloc[i, : ].tolist())
        train.append(instance)

    return train

def create_network_structure(network, train):
    with open(network) as network:
        lambda_val = float(network.readline().split()[0])
        neurons = []
        neurons.append(len(train[0][0])) # considera que a primeira camada tem número de neurônios igual a número de atributos do dataset (menos o atributo a ser predito)
        for line in network:
            neurons.append(int(line))
        s = {
            'lambda' : lambda_val,
            'neurons' : neurons
        }
    return s

def create_initial_weights(s):
    neurons = s['neurons']
    w = []
    for l in range(len(neurons)-1):
        layer = []
        cols = neurons[l]+1
        rows = neurons[l+1]
        for row in range(rows):
            row = []
            for col in range(cols):
                row.append(uniform(-1, 1))
            layer.append(row)
        layer = np.matrix(layer)
        w.append(layer)
    return w

def main():
    parser = argparse.ArgumentParser(description="Neural network")

    parser.add_argument('-n', "--network_structure", required=True, type=str,
                        help="name of the file with the structure of the neural network")
    
    parser.add_argument('-d', "--dataset", required=True, type=str,
                        help="name of the file with the dataset to train the neural network")

    parser.add_argument('-w', "--initial_weights", type=str, default=None,
                        help="name of the file with the initial_weights of the neural network")

    parser.add_argument("-v", "--numerical_verification", type=bool, default=False, 
                        help="execute the numerical verification (True or False)")

    parser.add_argument("-s", "--standard_normalization", type=bool, default=False, 
                        help="Normalize datasets using StandardScaler. If false, MinMaxScaler is used instead")

    parser.add_argument("-e", "--epsilon", type=float, default=0.000001, 
                        help="epsilon for the numerical verification")
    
    parser.add_argument("-p", "--predicted_index", type=int, default=0, 
                        help="index of the column to be predicted")

    parser.add_argument("--drop_column", type=int, default=None, 
                        help="index of a column to be dropped (an index column, for example)")

    parser.add_argument("--drop_row", type=int, default=None, 
                        help="index of a row to be dropped (a row with the names of the attributes, for example)")

    args = parser.parse_args()

    if args.initial_weights == None: # assume que, quando não são passados pesos iniciais, o dataset é no formato de um .csv normal
        train = create_train_set(args.dataset, args.predicted_index, None, None)
        s = create_network_structure(args.network_structure, train) # aqui o número de neurônios da primeira camada é calculado automaticamente, com base no número de atributos do dataset
        w = create_initial_weights(s)
    else: # caso contrário, o dataset está no formato da descrição do trabalho
        s, w, train = create_neural_network(args.network_structure, args.initial_weights, args.dataset) # aquio o número de neurônios da primeira camada é de acordo com o .txt, pra ficar de acordo com os exemplos deles

    print_network_parameters(s, w, train)
    network = nn.NeuralNetwork(s, w, args.epsilon)
    network.backpropagation(train)
    network.print_network()
    if(args.numerical_verification):
        numerical_gradients = network.compute_numerical_verification(train)
        network.print_numerical_verification(numerical_gradients)

if __name__ == "__main__":
    main()