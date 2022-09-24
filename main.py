import argparse
import datetime
from random import uniform

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import neural_network as nn


def get_categories(dataset, predicted_index):
    print(dataset)
    columnName = dataset.columns[predicted_index]  # nome da coluna a ser predita
    column = dataset[columnName]  # pega todos os dados da coluna
    columnValues = column.unique()  # separa cada valor único da coluna
    numberOfInstances = dict(column.value_counts())
    return columnValues, numberOfInstances


def create_neural_network(network, weights, dataset):
    with open(network) as network:
        lambda_val = float(network.readline().split()[0])
        print(lambda_val)
        neurons = []
        for line in network:
            neurons.append(int(line))
        s = {"lambda": lambda_val, "neurons": neurons}

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
            np_layer = np.matrix(layer)
            w.append(np_layer)

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
    return s, w, train


def print_network_parameters(s, w, train):
    print("Regularization parameter (lambda)=" + "{:.3f}".format(s["lambda"]))
    print("")
    print(
        "Initializing network with the following structure of neurons per layer: [",
        end="",
    )
    first = True
    for n in s["neurons"]:
        if first:
            first = False
            print(n, end="")
        else:
            print(" ", end="")
            print(n, end="")
    print("]")
    print("")
    for k in range(0, len(w)):
        print(
            "Initial theta"
            + str(k + 1)
            + " (the weights of each neuron, including bias, are stored in the rows):"
        )
        for i in range(0, len(w[k])):
            print("\t\t", end="")
            for j in range(0, w[k][i].size):
                print("{:.5f}".format(w[k][i, j]), end=" ")
            print("")
        print("")
    print("")
    print("Training set")
    exemplo = 1
    for instance in train:
        print("\tExample " + str(exemplo))
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


# assume que só tem um valor a ser previsto
def create_train_set(
    data, predicted_index, drop_col, drop_row, standard_normalization, dataset
):
    if drop_col is not None:
        data = data.drop(columns=[drop_col])
    if drop_row is not None:
        data = data.drop([drop_row])
    x = data.drop(columns=[predicted_index])
    y = data.drop(columns=[c for c in range(len(data.columns)) if c != predicted_index])
    train = []
    instances_atributes = []
    for i in range(len(data)):
        instances_atributes.append(x.iloc[i, :].tolist())

    scaler = MinMaxScaler()
    scaler.fit(instances_atributes)
    normalized_instances = scaler.transform(instances_atributes)

    for i in range(len(data)):
        instance = []
        instance.append(normalized_instances[i].tolist())
        if dataset == "wine":
            expected_output = np.zeros(3)
            expected_output[y.iloc[i, :] - 1] = 1
        elif dataset == "pima":
            expected_output = np.zeros(2)
            expected_output[y.iloc[i, :]] = 1
        elif dataset == "wdbc":
            expected_output = np.zeros(2)
            if y.iloc[i, :].tolist()[0] == "B":
                expected_output[0] = 1
            else:
                expected_output[1] = 1
        elif dataset == "iono":
            expected_output = np.zeros(2)
            if y.iloc[i, :].tolist()[0] == "b":
                expected_output[0] = 1
            else:
                expected_output[1] = 1
        else:
            expected_output = np.zeros(1)

        instance.append(expected_output.tolist())
        train.append(instance)

    return train


def create_network_structure(network, train):
    with open(network) as network:
        lambda_val = float(network.readline().split()[0])
        neurons = []
        # considera que a primeira camada tem número de neurônios igual
        # a número de atributos do dataset (menos o atributo a ser predito):
        neurons.append(len(train[0][0]))
        for line in network:
            neurons.append(int(line))
        s = {"lambda": lambda_val, "neurons": neurons}
    return s


def create_initial_weights(s):
    neurons = s["neurons"]
    w = []
    for i in range(len(neurons) - 1):
        layer = []
        cols = neurons[i] + 1
        rows = neurons[i + 1]
        for row in range(rows):
            row = []
            for _ in range(cols):
                row.append(uniform(-1, 1))
            layer.append(row)
        layer = np.matrix(layer)
        w.append(layer)
    return w


def cross_validation(
    dataset_file,
    predictionIndex,
    k,
    drop_col,
    drop_row,
    standard_normalization,
    network_structure,
    epsilon,
    dataset,
    max_iterations,
):
    if dataset == "pima":
        data = pd.read_csv(dataset_file, delimiter="\t", header=None)
    else:
        data = pd.read_csv(dataset_file, delimiter=",", header=None)
    categories, numberOfInstances = get_categories(data, predictionIndex)

    predictionColumnName = data.columns[predictionIndex]
    data_split_per_category = {}

    for category in categories:
        data_split_per_category[category] = data[data[predictionColumnName] == category]

    k_folds = [pd.DataFrame()] * k
    fscore = []

    # dividindo os folds estratificados
    for i in range(k):
        for category in categories:
            num_sample = numberOfInstances[category] // k
            sample = data_split_per_category[category].sample(n=num_sample)
            k_folds[i] = pd.concat([k_folds[i], sample])
            data_split_per_category[category] = data_split_per_category[category].drop(
                sample.index
            )

    # adicionando as instâncias que sobraram por categoria,
    # uma em cada fold (pra não ficar muito desparelha a quantidade total de instancias):
    instances_rest = pd.DataFrame()
    for category, data in data_split_per_category.items():
        instances_rest = pd.concat([instances_rest, data])

    fold_index = 0
    for i in range(len(instances_rest.index)):
        fold_index %= k
        k_folds[fold_index] = pd.concat([k_folds[fold_index], instances_rest.iloc[[i]]])
        fold_index += 1

    # rodando cross-validation de fato
    for test_fold_index, testing_data in enumerate(k_folds):
        # agrupando folds restantes em um dataframe só
        training_data = pd.DataFrame()
        for fold_index, fold in enumerate(k_folds):
            if fold_index != test_fold_index:
                training_data = pd.concat([training_data, fold])
        print("k-fold #%d" % test_fold_index)
        train = create_train_set(
            training_data,
            predictionIndex,
            drop_col,
            drop_row,
            standard_normalization,
            dataset,
        )
        test = create_train_set(
            testing_data,
            predictionIndex,
            drop_col,
            drop_row,
            standard_normalization,
            dataset,
        )
        # aqui o número de neurônios da primeira camada é calculado automaticamente,
        # com base no número de atributos do dataset:
        s = create_network_structure(network_structure, train)
        w = create_initial_weights(s)

        network = nn.NeuralNetwork(s, w, epsilon, max_iterations)
        network.backpropagation(train, test)

        # classifica cada instancia usando o ensemble que acabou de aprender
        results = []
        test = create_train_set(
            testing_data,
            predictionIndex,
            drop_col,
            drop_row,
            standard_normalization,
            dataset,
        )
        for instance in test:
            instance_classification = network.propagate(instance[0])
            results += [[instance[1], [x[0] for x in instance_classification.tolist()]]]

        fscore += [Fmeasure(results, categories)]

    print("F-Score average = %f" % np.mean(fscore))
    print("F-Score deviation = %f" % np.std(fscore))


def Fmeasure(results, categories, beta=1, score_mode="micro"):
    # valores acumulados para todas as classes,
    # a ser utilizado no caso de score_mode = micro média:
    VPac = VNac = FPac = FNac = 0
    all_precision = []
    # resultados de precisão e recall para cada classe,
    # a ser utilizado no caso de score_mode = macro média:
    all_recall = []
    # Tratamento multiclasse realizado independentemente da quantidade de classes
    for category in range(len(categories)):
        VP = VN = FP = FN = 0
        for res in results:
            r = []
            r += [np.argmax(res[0])]
            r += [np.argmax(res[1])]

            if r[0] == category and r[1] == category:
                VP += 1
            elif r[0] == category and r[1] != category:
                FN += 1
            elif r[0] != category and r[1] == category:
                FP += 1
            else:
                VN += 1

        if score_mode == "macro":
            # Evitar divisão por zero
            if VP + FP == 0:
                precision = 0
            else:
                precision = VP / (VP + FP)
            if VP + FN == 0:
                recall = 0
            else:
                recall = VP / (VP + FN)

            all_precision += [precision]
            all_recall += [recall]
        else:
            VPac += VP
            VNac += VN
            FPac += FP
            FNac += FN

    if score_mode == "micro":
        precision = VPac / (VPac + FPac)
        recall = VPac / (VPac + FNac)
    else:  # macro
        precision = np.mean(all_precision)
        recall = np.mean(all_recall)

    fscore = (
        (1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    )

    return fscore


def main():
    parser = argparse.ArgumentParser(description="Neural network")

    parser.add_argument(
        "-n",
        "--network_structure",
        required=True,
        type=str,
        help="name of the file with the structure of the neural network",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="{wine | pima | iono | <complete path to file>}",
    )

    parser.add_argument(
        "-w",
        "--initial_weights",
        type=str,
        default=None,
        help="name of the file with the initial_weights of the neural network",
    )

    parser.add_argument(
        "-v",
        "--numerical_verification",
        type=bool,
        default=False,
        help="execute the numerical verification (True or False)",
    )

    parser.add_argument(
        "-s",
        "--standard_normalization",
        type=bool,
        default=False,
        help="Normalize datasets using StandardScaler. If false, MinMaxScaler is used instead",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.000001,
        help="epsilon for the numerical verification",
    )

    parser.add_argument(
        "-p",
        "--predicted_index",
        type=int,
        default=0,
        help="index of the column to be predicted",
    )

    parser.add_argument(
        "--drop_column",
        type=int,
        default=None,
        help="index of a column to be dropped (an index column, for example)",
    )

    parser.add_argument(
        "--drop_row",
        type=int,
        default=None,
        help="index of a row to be dropped (a row with the names of the attributes, for example)",
    )

    parser.add_argument(
        "-k",
        "--folds_number",
        type=int,
        default=10,
        help="the number of folders to divide the dataset in cross-validation",
    )

    parser.add_argument(
        "-i",
        "--max_iterations",
        type=int,
        default=1,
        help="maximum number of times the backpropagation can be executed before being stopped",
    )

    args = parser.parse_args()

    if args.initial_weights is None:
        # assume que, quando não são passados pesos iniciais,
        # o dataset é no formato de um .csv normal
        if args.dataset == "wine":
            dataset = "wine"
            dataset_file = "datasets/wine.data"
        elif args.dataset == "pima":
            dataset = "pima"
            dataset_file = "datasets/pima.tsv"
        elif args.dataset == "iono":
            dataset = "iono"
            dataset_file = "datasets/ionosphere.data"
        elif args.dataset == "wdbc":
            dataset = "wdbc"
            dataset_file = "datasets/wdbc.data"
        else:
            print("Esse dataset não está configurado")
            exit()
        train = cross_validation(
            dataset_file,
            args.predicted_index,
            args.folds_number,
            None,
            None,
            args.standard_normalization,
            args.network_structure,
            args.epsilon,
            dataset,
            args.max_iterations,
        )

    else:  # caso contrário, o dataset está no formato da descrição do trabalho
        date = datetime.datetime.now()
        date = date.strftime("%d-%m-%Y_%H-%M-%S")
        # o número de neurônios da primeira camada é de acordo com o .txt,
        # pra ficar de acordo com os exemplos deles:
        s, w, train = create_neural_network(
            args.network_structure, args.initial_weights, args.dataset
        )
        print_network_parameters(s, w, train)
        network = nn.NeuralNetwork(s, w, args.epsilon, 1)
        backpropagation_gradients = network.backpropagation_with_prints(train)
        network.output_backpropagation(
            backpropagation_gradients, "output/" + date + "_backpropagation.txt"
        )
        network.print_network()
        if args.numerical_verification:
            numerical_gradients = network.compute_numerical_verification(train)
            network.print_numerical_verification(numerical_gradients)
            network.output_numerical_verification(
                numerical_gradients, "output/" + date + "_numerical_verification.txt"
            )


if __name__ == "__main__":
    main()
