import numpy as np
import math

def g(x):
    return np.matrix([1 / ( 1 + math.exp( -i ) ) for i in x]).transpose()

class NeuralNetwork:
    def __init__(self, struct, weights):
        self.thetas = weights
        self.struct = struct
        self.layers = len(self.struct['neurons'])
        
    
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
    
    
    # obtém saídas correspondentes a uma entrada
    def propagate (self, instance):
        # adiciona bias
        layer_entry = np.matrix([1] + instance).transpose() # valores de entrada para a camada sendo 
                                                            # a entrada da rede na primeira camada ou
                                                            # a ativação das camadas anteriores para 
                                                            # as demais camadas.
        print ("Entrada da rede para a camada 0: \n" + str(layer_entry))
        for layer in range(len(self.thetas) - 1):                    # para cada matriz de pesos theta entre duas camadas
            layer_entry = g(self.thetas[layer] * layer_entry)
            layer_entry = np.matrix([[1]] + layer_entry.tolist())   # adiciona neurônio de bias
            print ("Entrada da camada " + str(layer + 1) + ": \n" + str(layer_entry))
              
        

        return g(self.thetas[self.layers - 2] * layer_entry)         # (-2) pois existe uma tabela de thetas a menos que camada 
    

s = {
    'lambda' : 0.25,
    'neurons' : [2, 3, 1] # não conta o neurônio de bias, logo a segunda 
                          # camada terá 4 neurônios, isso não ateta a camada de
                          # entrada nem a de saída
}

w = [
        # layer 1 ( 3 x 3 ) # primeira coluna são os pesos do bias
        np.matrix([ 
            [0.1, 0.5, 0.8],
            [0.9, 0.1, 0.7],
            [0.4, 0.6, 0.2]
        ]),
        # layer 2 ( 1 x 4 ) # precisa considerar o bias, seu peso corresponde a primeira coluna
        np.matrix(
            [[0.2, 0.3, 0.6, 0.4]]
        )
    ]

inst = [2, 5]

nn = NeuralNetwork(s, w)
nn.print_network()

print ("Propagando valores na rede para a entrada: " + str(inst))
print ("Saída da rede: \n" + str(nn.propagate(inst)))

