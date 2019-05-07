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

Input file needed: node_features_train.csv, link.csv. You can change the link file for different evaluations.

- Example Command for learning embeddings is: 
```
cd src
python Graphsage.py --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt --save_emb 1 --save_emb_file ../data/cora/embed/graphsage_vec_lp.txt
```

- Here, you can have a general idea of the model's performance by inputting the evaluation files eval_rel.txt or eval/label.txt, or you can also input fake ones, and evaluate the learned embeddings more formally as following.

- Example Command for Node Classification & Example Command for Label Propagation respectively: 
```
cd src
python eval.py --type nc --embedding_file ../data/cora/embed/graphsage_vec_nc.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/labels.txt
python eval.py --type lp --embedding_file ../data/cora/embed/graphsage_vec_lp.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt
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
cd src
python eval.py --type nc --embedding_file ../data/cora/embed/tadw_cora_vec.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/labels.txt
python eval.py --type lp --embedding_file ../data/cora/embed/tadw_cora_vec.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt
```

3. #### GAT

- Code from ```Xingyu Fu```
- Contact: ```xingyuf2@illinois.edu```

- Example Command for learning embeddings is: 
```
cd src
python GAT.py --dataset ../data/cora/ --eval_file ../data/cora/eval/labels.txt --save_emb 1 --save_emb_file ../data/cora/embed/gat_vec_nc.txt --sample_size 2000
```

- Here, you can have a general idea of the model's performance by inputting the evaluation files eval_rel.txt or eval/label.txt, or you can also input fake ones, and evaluate the learned embeddings more formally as following.

- Example Command for Node Classification & Example Command for Label Propagation respectively: 
```
cd src
python eval.py --type nc --embedding_file ../data/cora/embed/gat_vec_nc.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/labels.txt
python eval.py --type lp --embedding_file ../data/cora/embed/gat_vec_lp.txt --dataset ../data/cora/ --eval_file ../data/cora/eval/rel.txt
```


4. #### planetoid

- Code from ```Haonan Wang```. Contact: ```haonanw2@illinois.edu```

- Unzip the zip folder first. 
- Change your input file and output embedding file in ```planetoid_edge/test_ind.py```. Notice that ```rel_train_path``` has data in form of ```node_1  node_2  link_label```
- Then perform evaluation on the learned embeddings as previous commands for TADW.

Example Command:
```
cd src/planetoid_edge
python test_ind.py --rel_train_path ../../data/cora/rel.txt --rel_test_path ../../data/cora/eval/rel.txt --embedding_path ../../data/cora/embed/planetoid_vec_lp.txt
```

5. #### CANE

- Code from ```https://github.com/thunlp/CANE.git```


- Unzip the zip folder first. Change your input file and output embedding file in ```CANE/code/train.py```. Then perform evaluation on the learned embeddings as previous commands for TADW.

Example Command:
```
cd src/CANE/code
python train.py
```
and in ```train.py```, write 
```
save_embed_file = '../../data/cora/embed/cane_vec_lp.txt'
```

6. #### STNE

- Code comes from original authors.

- You can change input in src/STNE.py and output the embedding to designated location. Then perform evaluation on the learned embedding as previous commands for TADW.

Example Command:
```
cd src
python STNE.py
```