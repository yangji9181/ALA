# Graph_Sample

- This repo contains baseline instructions for unsupervised network embedding methods.
- Author: Xingyu Fu
- Contact: xingyuf2@illinois.edu

## Data Used
- dblp
- cora

## Example Inputs
Examples are shown in folder ```data/cora/```.

`eval/label.txt` contains labels of graph nodes for evaluation.
`eval/rel.txt` contains graph links that are extracted to be test data.
`edgelist.txt` are input graph links for embedding learning.

## Evaluation Methods
#### Link Prediction
Learned embeddings for every two nodes are concatenated and passed through a set of layers: fully connected layer + RELU + fully connected layer, and returns accuracy and f1 score.

#### Node Classification
Learned embeddings for a graph node is passed into a multi-label classifier. Specifically, it's a OneVSRest Classifier using logistic regression.


## Supervised methods -> Unsupervised setting
- Node classification loss in the supervised methods has been changed to link prediction loss using the aformentioned prediction structure.

## Dependencies


This project is based on ```python>=3.6``` and ```pytorch>=0.4```. 


## Baselines:
1. #### GraphSage
---------------

- Code from ```https://github.com/JieyuZ2/RELEARN```

Input file needed: node_features_train.csv, link.csv, eval_rel.txt or eval/label.txt. 

Example Command for Label Propagation: 
```
cd src
python graphsage.py --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt

python graphsage.py --dataset ../data/dblp/ --eval_file ../data/dblp/eval/rel.txt
```

Example Command for Node Classification: 
```
cd src
python graphsage.py --dataset ../data/cora/ --eval_file ../data/cora/eval/label.txt
```

2. #### TADW
---------------

- Code from ```https://github.com/thunlp/OpenNE```

- After installing OpenNE, example Command for learning embeddings is: 
```
python -m openne --method tadw --input ../data/cora/edgelist.txt --graph-format edgelist --feature-file ../data/cora/features.txt --label-file ../data/cora/labels.txt --representation-size 100 --epochs 200 --output ../data/cora/embed/tadw_cora_vec.txt --clf-ratio 0.1
```
Here, the input label file is not important and output is path to the learned embedding. Then, we perform evaluation on learned embeddings.


- Example Command for Node Classification & Example Command for Label Propagation respectively: 
```
python eval.py --type nc --embedding_file ../data/cora/embed/tadw_cora_vec.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/labels.txt
python eval.py --type lp --embedding_file ../data/cora/embed/tadw_cora_vec.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt
```

3. #### GAT

- Code from ```Xingyu Fu```
- Contact: ```xingyuf2@illinois.edu```

Example Command for Label Propagation: 
```
cd src
python GAT.py --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt --sample_size 2000
```

Example Command for Node Classification: 
```
cd src
python GAT.py --dataset ../data/cora/ --eval_file ../data/cora/eval/label.txt
```

4. #### planetoid

- Code from ```Haonan Wang```
- Contact: ```haonanw2@illinois.edu```

Example Command:
```
cd planetoid_edge
sh train_planetoid.sh
```

5. #### CANE

- Code from ```https://github.com/thunlp/CANE.git```

Example Command:
```
python3 run.py --dataset cora --gpu 0  --ratio 0.55 --rho 1.0,0.3,0.3
```

6. #### STNE

- Code from original authors, and contained in zip.

Example Command:
```
python main.py
```