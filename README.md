# Improving Attention Mechanism in Graph Neural Networks via Cardinality Preservation (IJCAI-2020)

<p align="center">
<img src="https://github.com/zetayue/CPA/blob/master/CPA.png?raw=true">
</p>

Code for the Cardinality Preserved Attention (CPA) model proposed in our [paper](https://www.ijcai.org/Proceedings/2020/0194.pdf). 

## Requirements
* CUDA 10.2
* Python 3.6.9
* Pytorch 1.6.0
* Pytorch Geometric 1.6.1
* Pytorch Scatter 2.0.5
* Pytorch Sparse 0.6.7
* NumPy
* scikit-learn

When you have an environment with Python 3.6.9 and CUDA 10.2, the other dependencies can be installed with:
```
pip install -r requirements.txt
```
## How to run
Unzip the data file:
```
unzip data.zip
```
Train and test our model:
```
python main.py 
```
Optional arguments:
```
  --dataset         name of dataset
  --mod             model to be used: origin, additive, scaled, f-additive, f-scaled
  --seed            random seed
  --epochs          number of epochs to train
  --lr              initial learning rate
  --wd              weight decay value
  --n_layer         number of hidden layers
  --hid             size of input hidden units
  --heads           number of attention heads
  --dropout         dropout rate
  --alpha           alpha for the leaky_relu
  --kfold           number of kfold
  --batch_size      batch size
  --readout         readout function: add, mean
```
In our paper, the MUTAG, PROTEINS, ENZYMES, NCI1, REDDIT-BINARY and REDDIT-MULTI-5K datasets on https://chrsmrrs.github.io/datasets/docs/datasets/ are used. The other datasets listed on the website can also be used by directly changing the name of the dataset in '--dataset'. When you run the code, the needed dataset will be automatically downloaded and processed.
## Cite
If you found this model and code are useful, please cite our paper:
```
@inproceedings{ijcai2020-194,
  title     = {Improving Attention Mechanism in Graph Neural Networks via Cardinality Preservation},
  author    = {Zhang, Shuo and Xie, Lei},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {1395--1402},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/194},
  url       = {https://doi.org/10.24963/ijcai.2020/194},
}
```