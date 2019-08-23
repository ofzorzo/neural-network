# neural-network
Neural network implemented for UFRGS' machine learning course. The neural network was used to predict data from the ionosphere.data, pima.tsv, wdbc.data and wine.data datasets. There's a nice report included, where the results obtained are analyzed, but it's in Portuguese.

## Execution examples

Default datasets:
```
python main.py -n datasets/wine.txt -d wine -i 100
python main.py -n datasets/pima.txt -d pima -p 8 -i 100
python main.py -n datasets/iono.txt -d iono -p 34 -i 100
python main.py -n datasets/wdbc.txt -d wdbc -p 1 -i 100 --drop_column 0
```

Arbitrary datasets:
```
python main.py -n examples/network.txt -d examples/dataset.txt -w examples/initial_weights.txt
```
