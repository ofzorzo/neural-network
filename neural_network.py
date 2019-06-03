import numpy as np
import math

def g(x):
    return np.matrix([1 / ( 1 + math.exp( -i ) ) for i in x]).transpose()

def element_wise_mult(m1, m2):
    ans = []
    
    if m1.shape == m2.shape:
        for i in range(m1.shape[0]):
            ans.append([m1[i, j] * m2[i, j] for j in range(m1.shape[1])])
        return np.matrix(ans)
    else:
        raise Exception ("Matrix must be the same shape to perform element-wise multiplication")


class NeuralNetwork:
    def __init__(self, struct, weights):
        self.thetas = weights
        self.struct = struct
        self.a = self.init_activations()
        self.deltas = self.init_activations()
        self.layers = len(self.struct['neurons'])
        self.alpha = 0.1

    def init_activations(self):
        activations = []
        for layer_size in self.struct['neurons']:
            activations.append([0 for i in range (layer_size)] )
        return activations

    
    
    
    def backpropagation (self, train):
        Grad = [ [] for i in range(self.layers - 1)] # gradiente acumulado dos exemplos
        print ("Calculando erro da rede: ")
        for sample in train:
            print ("\n")
            print ("Processando entrada " + str(sample[0]))
            predicted = self.propagate(sample[0])
            print("")
            for l in range(len(self.a)):
                print ("Ativação da camada " + str(l) + ":\n" + str(self.a[l].tolist()))
            print("")
            print ("Predicted = \n" + str(predicted))
            print ("Expected = \n" + str(sample[1]))
            expected = np.matrix(sample[1]).transpose()
            self.deltas[self.layers-1] = predicted - expected
            for k in range(self.layers-2, 0, -1):
                temp = (self.thetas[k].transpose() * self.deltas[k+1])
                temp = element_wise_mult(temp, self.a[k])
                temp = element_wise_mult(temp, 1 - self.a[k])
                self.deltas[k] = temp[1:] # não conta o delta do bias
            
            print("")
            for l in range(len (self.deltas) - 1, 0, -1):
                print ("Deltas " + str(l + 1) + ":\n" + str(self.deltas[l])) 
            
            grad_inp = [ [] for i in range(self.layers - 1) ]  # gradiente para um exemplo
            for k in range(self.layers-2, -1, -1):

                grad_inp[k] = self.deltas[k+1] * self.a[k].transpose()
                #acumula gradiente
                if len(Grad[k]) == 0: # primeiro exemplo
                    Grad[k] = grad_inp[k]
                else:
                    Grad[k] = [Grad[k][i] + grad_inp[k][i]  for i in range(len(grad_inp[k]))] 
            
            print("")
            for l in range(len(grad_inp)-1, -1, -1):
                print ("Gradientes para theta" + str(l+1) + " da entrada:\n" + str(grad_inp[l]))
            
        print("")
        
        for k in range(self.layers-2, -1, -1):
            P = self.struct['lambda'] * self.thetas[k]
            # faz primeira coluna ficar em zeros -> não penalizar bias
            for i in range(len(P)):
                P[i,0] = 0
            Grad[k] = (1/len(train)) * np.array([Grad[k][i] + P[i] for i in range(len(P))])
        
        for l in range(len(Grad)):
            print ("Gradientes acumulados (com regularização) para theta" + str(l+1) + ":\n" + str(Grad[l]))
        

        for k in range(self.layers-2, -1 , -1):
            self.thetas[k] = np.matrix(self.thetas[k]) - np.matrix(self.alpha * Grad[k])

        for t in range (len(self.thetas)):
            print ("Novo theta " + str(t+1) +" com alpha " + str(self.alpha) + ":\n" + str(self.thetas[t]))
            

    # obtém saídas correspondentes a uma entrada
    def propagate (self, instance):
        # adiciona bias
        self.a[0] = np.matrix([1] + instance).transpose() # valores de entrada para a camada sendo 
                                                            # a entrada da rede na primeira camada ou
                                                            # a ativação das camadas anteriores para 
                                                            # as demais camadas.
        for layer in range(len(self.thetas) - 1):                    # para cada matriz de pesos theta entre duas camadas
            self.a[layer + 1] = g(self.thetas[layer] * self.a[layer])
            self.a[layer + 1] = np.matrix([[1]] + self.a[layer + 1].tolist())   # adiciona neurônio de bias
               
        self.a[self.layers-1] = g(self.thetas[self.layers - 2] * self.a[self.layers - 2])         # (-2) pois existe uma tabela de thetas a menos que camada 
        return self.a[self.layers-1]
    
    def print_network (self):
        print ("Rede com " + str(self.layers) +  " camadas:\n") 
        layer = 0
        for n in self.struct['neurons']:
            print ("Camada " + str(layer) + " possui " + str(n) +  " neuronios")
            layer+=1
        print ("\n\nPesos : ")
        for theta in range(len(self.thetas)):
            print ("Pesos entre camada " + str(theta) + " e camada " + str(theta + 1) + ":")
            print (self.thetas[theta])
        
        print ("\n\n")

s = {
    'lambda' : 0.25,
    'neurons' : [2, 4, 3, 2] # não conta o neurônio de bias, logo a segunda 
                             # camada terá 4 neurônios, isso não ateta a camada de
                             # entrada nem a de saída
}

w = [
        np.matrix([ 
            [0.42, 0.15, 0.40],
            [0.72, 0.10, 0.54],
            [0.01, 0.19, 0.42],
            [0.30, 0.35, 0.68]
        ]),
        np.matrix([
            [0.21, 0.67, 0.14, 0.96, 0.87],
            [0.87, 0.42, 0.20, 0.32, 0.89],
            [0.03, 0.56, 0.80, 0.69, 0.09]
        ]),
        np.matrix([
            [0.04, 0.87, 0.42, 0.53],
            [0.17, 0.10, 0.95, 0.69]
        ])
    ]

train = [
    [[0.32, 0.68],[0.75, 0.98]],
    [[0.83, 0.02],[0.75, 0.28]]
]

nn = NeuralNetwork(s, w)

nn.backpropagation(train)
