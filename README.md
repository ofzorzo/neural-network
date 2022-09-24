# neural-network
This project consists of a neural network implemented from scratch for UFRGS' machine learning course. The neural network is trained with backpropagation and its structure can be personalized by the user.

The neural network was trained with four fixed datasets (used to predict data from the ionosphere, pima, wdbc and wine) and its performance was analyzed.

## Installation

Poetry is used to manage the dependencies of this project. Therefore, installing it is as easy as cloning this repository and then running:
```
poetry install
```

## Usage

### Arbitrary datasets

For arbitrary datasets, you must pass the following arguments when calling the program:

- `-n`: file containing the structure of the neural network. The first line stores the value of the lambda parameter used for regularization and the subsequent lines store the number of neurons in each corresponding layer of the network;
- `-d`: file containing the dataset used to train the neural network;
- `-w`: file containing the initial weights of the neural network.

Besides these three mandatory arguments, there are additional optional arguments that can be used. You can check them out by passing the `-h` argument to the program.

#### Execution example:
```
python main.py -n examples/network.txt -d examples/dataset.txt -w examples/initial_weights.txt
```
### Analyzed datasets

The usage is basically the same as for arbitrary datasets. However, you can omit the `-n` argument: we included the best network structure we found for each of the four datasets and they are loaded automatically.

#### Execution example:
```
python main.py -n datasets/wine.txt -d wine -i 100
python main.py -n datasets/pima.txt -d pima -p 8 -i 100
python main.py -n datasets/iono.txt -d iono -p 34 -i 100
python main.py -n datasets/wdbc.txt -d wdbc -p 1 -i 100 --drop_column 0
```

## Performance analysis
todo