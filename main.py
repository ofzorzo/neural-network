import argparse
import neural_network as nn
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description="Neural network")

    parser.add_argument('files', type=str, nargs=3,
                        help="list of the network structure, initial weights and dataset files (in that order)")

    parser.add_argument("-v", "--numerical_verification", type=bool, default=False, 
                        help="execute the numerical verification (True or False)")

    parser.add_argument("-e", "--epsilon", type=float, default=0.000001, 
                        help="epsilon for the numerical verification")

    args = parser.parse_args()

    s, w, train = create_neural_network(args.files[0], args.files[1], args.files[2])
    print_network_parameters(s, w, train)
    network = nn.NeuralNetwork(s, w, args.epsilon)
    #network.backpropagation(train)
    #network.print_network()
    if(args.numerical_verification):
        numerical_gradients = network.compute_numerical_verification(train)
        network.print_numerical_verification(numerical_gradients)

if __name__ == "__main__":
    main()