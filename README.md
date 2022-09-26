# neural-network
This project consists of a neural network implemented from scratch for UFRGS' machine learning course in 2019. The neural network was trained with backpropagation and its structure can be personalized by the user.

## Installation

[Poetry](https://github.com/python-poetry/poetry) is used to manage the dependencies of this project. Therefore, installing it is as easy as cloning this repository with:

```
git clone git@github.com:ofzorzo/neural-network-ufrgs.git
```

And then running:
```
poetry install
```

## Usage

This program has two modes: **arbitrary** and **fixed** datasets.

### Arbitrary datasets

The first mode uses arbitrary datasets and outputs the network's gradients found by backpropagation. This mode was mainly used to check the correctness of the neural network and the backpropagation algorithm; therefore, a numerical checking tool is also provided.

To use this mode, you must pass the following arguments when calling the program:

- `-n STRUCTURE`: file containing the structure of the neural network. The first line stores the value of the lambda parameter used for regularization and the subsequent lines store the number of neurons in each corresponding layer of the network;
- `-d DATASET`: file containing the dataset used to train the neural network;
- `-w WEIGHTS`: file containing the initial weights of the neural network.

If you wish to perform a numerical gradient checking to confirm the correctness of the backpropagation algorithm, you must pass the `-v True` argument.

#### Execution example:
```
python main.py -n examples/network.txt -d examples/dataset.txt -w examples/initial_weights.txt -v True
```
### Fixed datasets

The neural network was trained with four fixed datasets ([ionosphere](https://archive.ics.uci.edu/ml/datasets/Ionosphere), [wdbc](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [wine](https://archive.ics.uci.edu/ml/datasets/wine) and pima) and its performance was analyzed for each case.

The usage is basically the same as for arbitrary datasets. The `-w` argument, however, can be omitted, since the weights are randomly initialized. Furthermore, there are additional optional arguments that can be used. You can check them out by passing the `-h` flag to the program.

#### Execution example:
```
python main.py -n datasets/wine.txt -d wine -i 100
python main.py -n datasets/pima.txt -d pima -p 8 -i 100
python main.py -n datasets/iono.txt -d iono -p 34 -i 100
python main.py -n datasets/wdbc.txt -d wdbc -p 1 -i 100 --drop_column 0
```

Notice that [wine.txt](datasets/wine.txt), [pima.txt](datasets/pima.txt), [iono.txt](datasets/iono.txt) and [wdbc.txt](datasets/wdbc.txt) are the best network structures we found for each respective dataset while analyzing the performance of the network. The report in which the performance was analyzed is included [here](/report/report.pdf). Since it was a college project, it's written in Portuguese.
