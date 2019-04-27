# graph_sample

This repo contains baseline informations for unsupervised network embedding methods.

#Data Used
- dblp
- cora

# Required Inputs
---------------
node_features.csv, link.csv, eval.txt

# Dependencies
---------------

This project is based on ```python>=3.6``` and ```pytorch>=0.4```. 

# Baselines:
1. ###GraphSage
---------------

- Code from ```https://github.com/JieyuZ2/RELEARN```

Input file needed: node_features_train.csv, link.csv, eval_rel.txt or eval/label.txt. 

Example Command for Label Propagation: 
```
python graphsage.py --dataset ../../data/cora/ --eval_file ../../data/cora/eval/rel.txt

python graphsage.py --dataset ../../data/dblp/ --eval_file ../../data/dblp/eval/rel.txt
```

Example Command for Node Classification: 
```
python graphsage.py --dataset ../../data/cora/ --eval_file ../../data/cora/eval/label.txt
```

2. #### TADW
---------------

- Code from ```https://github.com/thunlp/OpenNE```


Example Command for Node Classification: 
```
python -m openne --method tadw --label-file ../data/cora/labels.txt --input ../data/cora/edgelist.txt --graph-format edgelist --feature-file ../data/cora/features.txt --representation-size 100 --epochs 200 --output ../data/cora/embed/tadw_cora_vec.txt --clf-ratio 0.1

python -m openne --method tadw --input ../data/dblp/edgelist.txt --graph-format edgelist --feature-file ../data/dblp/features.txt --label-file ../data/dblp/eval/fake-label.txt --representation-size 100 --epochs 500 --output ../data/dblp/embed/tadw_dblp_vec.txt
```

Example Command for Label Propagation: 
```
python eval.py --type lp --embedding_file ../data/cora/embed/tadw_cora_vec.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt

python eval.py --type lp --embedding_file ../data/dblp/embed/tadw_dblp_vec.txt --dataset ../data/dblp/ --eval_file ../data/dblp/eval/rel.txt
```

3. #### GAT

- Author: Xingyu Fu
- Contact: xingyuf2@illinois.edu

Example Command for Label Propagation: 
```
python GAT.py --dataset ../../data/cora/ --eval_file ../../data/cora/eval/rel.txt --sample_size 2000

python graphsage.py --dataset ../../data/dblp/ --eval_file ../../data/dblp/eval/rel.txt
```

Example Command for Node Classification: 
```
python graphsage.py --dataset ../../data/cora/ --eval_file ../../data/cora/eval/label.txt
```
