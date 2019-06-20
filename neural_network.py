import numpy as np
import math
import copy

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
    def __init__(self, struct, weights, epsilon, max_iterations):
        self.thetas = weights
        self.struct = struct
        self.a = self.init_activations()
        self.deltas = self.init_activations()
        self.layers = len(self.struct['neurons'])
        self.alpha = 0.8
        self.beta = 0.9
        self.max_iterations = max_iterations
        self.batch_size = 32
        self.thetas_numerical = copy.deepcopy(self.thetas)
        self.a_numerical = copy.deepcopy(self.a)        
        self.epsilon = epsilon

    def init_activations(self):
        activations = []
        for layer_size in self.struct['neurons']:
            activations.append([0 for i in range (layer_size)] )
        return activations

    def stop_condition(self, i, prev_thetas):
        if i == 0: # primeira iteração, precisa dessa condição pois o valor de prev_theta será None
            return False
        else:
            # realiza o backpropagation até atingir o limite máximo de iterações ou até os thetas pararem
            # de mudar
            #print(i)
            return i >= self.max_iterations or np.array_equal(self.thetas, prev_thetas)

    def cost_function(self, train):
        J = 0
        for sample in train:
            f_x = self.propagate(sample[0])
            y = sample[1]
            J_i = [-y[i] * np.log(f_x[i]) - (1 - y[i]) * np.log(1-f_x[i]) for i in range(len(f_x))]
            J += np.sum(J_i)
        J = J / len(train)
        S = 0
        for layer in range(len(self.thetas)):
            for row in range(len(self.thetas[layer])):
                for col in range(len(self.thetas[layer][row].transpose())):
                    S+= (self.thetas[layer][row, col]**2)
        S = (self.struct['lambda'] / (2 * len(train))) * S
        return int(S) + J
    # Considera um conjunto de treinamento formado por uma lista de pares do tipo:
    # (entradas, saídas esperadas), onde ambos os elementos são listas de valores inteiros
    def backpropagation_with_prints(self, train):
        itera = 0
        prev_theta = None
        while (not self.stop_condition(itera, prev_theta)):
            prev_theta = self.thetas[:] # copia por valor a matriz dos thetas
            itera+=1
            print ("Iteration " + str(itera) + ":") 
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
        
            print("\n--------------------------------------------")
        return Grad
    
    # Considera um conjunto de treinamento formado por uma lista de pares do tipo:
    # (entradas, saídas esperadas), onde ambos os elementos são listas de valores inteiros
    def backpropagation(self, train, test):
        itera = 0
        prev_theta = None
        while (not self.stop_condition(itera, prev_theta)):
            prev_theta = self.thetas[:] # copia por valor a matriz dos thetas
            itera+=1
            print ("Iteration " + str(itera) + ":") 
            print ("Train cost = " + str(self.cost_function(train)))
            print ("Test cost = " + str(self.cost_function(test)))
            Grad = [ [] for i in range(self.layers - 1)] # gradiente acumulado dos exemplos
            Grad_total = [ [] for i in range(self.layers - 1)] # gradiente acumulado dos exemplos
            num_samples = 0
            for sample in train:
                predicted = self.propagate(sample[0])
                expected = np.matrix(sample[1]).transpose()
                self.deltas[self.layers-1] = predicted - expected
                for k in range(self.layers-2, 0, -1):
                    temp = (self.thetas[k].transpose() * self.deltas[k+1])
                    temp = element_wise_mult(temp, self.a[k])
                    temp = element_wise_mult(temp, 1 - self.a[k])
                    self.deltas[k] = temp[1:] # não conta o delta do bias
                
                grad_inp = [ [] for i in range(self.layers - 1) ]  # gradiente para um exemplo
                for k in range(self.layers-2, -1, -1):
                    grad_inp[k] = self.deltas[k+1] * self.a[k].transpose()
                    #acumula gradiente
                    if len(Grad_total[k]) == 0: # primeiro exemplo
                        Grad_total[k] = np.zeros(np.shape(grad_inp[k]))
                    if len(Grad[k]) == 0: # primeiro exemplo
                        Grad[k] = grad_inp[k]
                    else:
                        Grad[k] = [Grad[k][i] + grad_inp[k][i]  for i in range(len(grad_inp[k]))] 
                
                num_samples +=1

                if num_samples == self.batch_size:
                    for k in range(self.layers-2, -1, -1):
                        P = self.struct['lambda'] * self.thetas[k]
                        # faz primeira coluna ficar em zeros -> não penalizar bias
                        for i in range(len(P)):
                            P[i,0] = 0
                        Grad[k] = (1/len(train)) * np.array([Grad[k][i] + P[i] for i in range(len(P))])
                    
                    for k in range(self.layers-2, -1, -1):
                        Grad_total[k] = self.beta * np.matrix(Grad_total[k]) + np.matrix(Grad[k])
                        self.thetas[k] = np.matrix(self.thetas[k]) - np.matrix(self.alpha * Grad_total[k])
                        # self.thetas[k] = np.matrix(self.thetas[k]) - np.matrix(self.alpha * Grad[k])

                    Grad = [ [] for i in range(self.layers - 1)] # gradiente acumulado dos exemplos
                    num_samples = 0

            print("\n--------------------------------------------")
        return Grad

    # obtém saídas correspondentes a uma entrada
    def propagate(self, instance):
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

    def compute_numerical_verification(self, train):
        print("Rodando verificacao numerica de gradientes (epsilon=" + "{:.10f}".format(self.epsilon) + ")")
        gradients = copy.deepcopy(self.thetas_numerical)
        for k in range(self.layers-2, -1, -1):
                #print(self.thetas_numerical[k])
                for i in range(0, len(self.thetas_numerical[k])):
                    for j in range(0, self.thetas_numerical[k][i].size):
                        #print(self.thetas_numerical[k][i,j])
                        Jp = 0
                        Jn = 0
                        reg_p, reg_n = self.regularized_terms(k, i, j)
                        for sample in train:
                            predicted_p = self.propagate_numerical(sample[0], self.thetas_numerical, k, i, j, "+")
                            predicted_n = self.propagate_numerical(sample[0], self.thetas_numerical, k, i, j, "-")
                            expected = np.matrix(sample[1]).transpose()
                            for out in range(0, len(expected)): # soma os erros das múltiplas saídas
                                Jp = Jp + ( -expected[out, 0]*(math.log(predicted_p[out, 0])) - (1-expected[out, 0])*(math.log(1-predicted_p[out, 0])) )
                                Jn = Jn + ( -expected[out, 0]*(math.log(predicted_n[out, 0])) - (1-expected[out, 0])*(math.log(1-predicted_n[out, 0])) )
                        Jp = Jp/len(train)
                        Jn = Jn/len(train)
                        if j>0: # não atualiza pesos do bias com o termo de regularização
                            Jp = Jp + (self.struct['lambda']/(2*len(train)))*reg_p
                            Jn = Jn + (self.struct['lambda']/(2*len(train)))*reg_n
                        gradients[k][i,j] = (Jp-Jn)/(2*self.epsilon)
        return gradients
    
    def regularized_terms(self, k, i, j):
        reg_p=0
        reg_n=0
        for k2 in range(self.layers-2, -1, -1):
            #print(self.thetas_numerical[k])
            for i2 in range(0, len(self.thetas_numerical[k2])):
                #print("termo ignorado = " + str(self.thetas_numerical[k2][i2,0]))
                for j2 in range(1, self.thetas_numerical[k2][i2].size): # começa em 1 para não somar os pesos de bias
                    #print("\ttermo atual = " + str(self.thetas_numerical[k2][i2,j2]))
                    if k2==k and i2==i and j2==j:
                        reg_n = reg_n + pow(self.thetas_numerical[k2][i2,j2]-self.epsilon, 2)
                        reg_p = reg_p + pow(self.thetas_numerical[k2][i2,j2]+self.epsilon, 2)
                    else:
                        reg_n = reg_n + pow(self.thetas_numerical[k2][i2,j2], 2)
                        reg_p = reg_p + pow(self.thetas_numerical[k2][i2,j2], 2)
        return reg_p, reg_n                        

    def propagate_numerical(self, instance, thetas, k, i, j, op):
        self.a_numerical[0] = np.matrix([1] + instance).transpose()
        if op == "+":
            thetas[k][i,j] = thetas[k][i,j] + self.epsilon
        elif op == "-":
            thetas[k][i,j] = thetas[k][i,j] - self.epsilon
        for layer in range(len(thetas) - 1):
            self.a_numerical[layer + 1] = g(thetas[layer] * self.a_numerical[layer])
            self.a_numerical[layer + 1] = np.matrix([[1]] + self.a_numerical[layer + 1].tolist())               
        self.a_numerical[self.layers-1] = g(thetas[self.layers - 2] * self.a_numerical[self.layers - 2])
        if op == "+":
            thetas[k][i,j] = thetas[k][i,j] - self.epsilon
        elif op == "-":
            thetas[k][i,j] = thetas[k][i,j] + self.epsilon
        return self.a_numerical[self.layers-1]
    
    def print_network(self):
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
    
    def print_numerical_verification(self, numerical_gradients):
        for k in range(0, len(numerical_gradients)):
            print("\tGradiente numerico de Theta" + str(k+1) + ":")
            for i in range(0, len(numerical_gradients[k])):
                print("\t\t", end="")
                for j in range(0, numerical_gradients[k][i].size):
                    print("{:.5f}".format(numerical_gradients[k][i, j]), end=" ")
                print("")
            print("")
        print("")
        print("--------------------------------------------")
    
    def output_numerical_verification(self, numerical_gradients, filename):
        f = open(filename, "w")
        first_layer = True
        for k in range(0, len(numerical_gradients)):
            if first_layer:
                first_layer = False
            else:
                f.write("\n")
            first_neuron = True
            for i in range(0, len(numerical_gradients[k])):
                if first_neuron:
                    first_neuron = False
                else:
                    f.write("; ")
                first_weight = True
                for j in range(0, numerical_gradients[k][i].size):
                    if first_weight:
                        f.write("{:.5f}".format(numerical_gradients[k][i, j]))
                        first_weight = False
                    else:
                        f.write(", " + "{:.5f}".format(numerical_gradients[k][i, j]))                    
        f.close()
    
    def output_backpropagation(self, gradients, filename):
        f = open(filename, "w")
        first_layer = True
        for layer in range(0, len(gradients)):
            if first_layer:
                first_layer = False
            else:
                f.write("\n")
            first_neuron = True
            for next in range(0, len(gradients[layer])): #escolhe linha da matriz, que representa o neuronio onde os pesos da linha vão chegar
                if first_neuron:
                    first_neuron = False
                else:
                    f.write("; ")
                first_weight = True
                for weight in range(0, gradients[layer][next].size):
                    if first_weight:
                        f.write( "{:.5f}".format(gradients[layer][next].item(weight)) )
                        first_weight = False
                    else:    
                        f.write( ", " + "{:.5f}".format(gradients[layer][next].item(weight)) )
        f.close()
