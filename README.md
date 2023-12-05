# PGTNet: A Process Graph Transformer Network for Remaining Time Prediction of Business Process Instances
This is the supplementary githob repository of the paper: "PGTNet: A Process Graph Transformer Network for Remaining Time Prediction of Business Process Instances".

Our approach consists of a data transformation from an event log to a graph dataset, and training a neural network based on the [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS) recipe (It is called the  **GPS repository** in the remaining of this README file). In order to start your experiments with PGTNet, the first thing to do is to clone both **GPS repository** and **PGTNet repository**.

**1. Set up a Python environement to work with GPS Graph Transformers:**

Based on the [instructions](https://github.com/rampasek/GraphGPS#python-environment-setup-with-conda), you need to set up a Python environement with Conda:
```
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

pip install PyYAML
pip install numpy
pip install pm4py
pip install scikit-learn

conda clean --all
```
Note that we included pip install commands required for working with event log data (e.g., [pm4py](https://pm4py.fit.fraunhofer.de/) library).

Once you setup a conda environement and installed all libraries required for woring with GPS Graph Transformers, copy the **PGTNet repository** into the main directory that is used for **GPS repository**. The structure should be like this:
```
GraphGPS
│
├── configs
├── graphgps
├── PGTNet
├── run
├── tests
├── unittests
├── main.py
└── README.md
```
**<a name="part1">1. Data preparation:</a>**
Converting event logs into graph datasets. For more information, see: [conversion directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion). We already uploaded generated graph datasets in [conversion/transformation directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/transformation) in this repository. Therefore, this step can be skipped if you are not intreseted in conducting more experiments with feature engineering. In this case, generated graph dataset are directly used in the second step to train and evaluaate GPS graph transformers.

**<a name="part2">2. Training and evaluation of GPS graph transformers:</a>**

_<a name="part2-1">2.1. Original implementation of the GPS graph transformers, and setting up a python environment:</a>_



_<a name="part2-2">2.2. Adjustments to the original implementation:</a>_

Once requirements of [2.1.](https://github.com/keyvan-amiri/GT-Remaining-CycleTime#part2-1) are met, the cloned version of the original implementation should be adjusted as per follows:

a. Replace `main.py` file in the root directory of the `GPS repository` by [the one with same name](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/main.py) in this repository. Here, the most important change is that a train mode called 'event-inference' is added to customize the inference step.

b. Move to the directory `/graphgps/loader` in the `GPS repository` and replace `master_loader.py` by [the one with same name](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/master_loader.py) in this repository. Here, the most important change is that several new dataset classes are added to handle graph representation of event logs.

c. Copy `GTeventlogHandler.py` from this repository and paste it in the directory `/graphgps/loader/dataset` in the `GPS repository`. [This file](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/GTeventlogHandler.py) includes multiple `InMemoryDataset` Pytorch Geometric classes. We created one seperate class for each event log.

d. Move to the directory `/graphgps/encoder` in the `GPS repository` and replace `linear_edge_encoder.py` by [the one with same name](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/linear_edge_encoder.py) in this repository. Additionally, you need to copy the [file](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/two_layer_linear_edge_encoder.py) `two_layer_linear_edge_encoder.py` from this repository and paste it in the same directory: `/graphgps/encoder` in the `GPS repository`. These two files are specifically designed for edge embedding in the remaining cycle time prediction problem.

_<a name="part2-3">2.3. Training a GPS graph transformer network:</a>_

Once requirements of [2.2.](https://github.com/keyvan-amiri/GT-Remaining-CycleTime#part2-2) are met, training the network is straightforward. Training is done using the relevant .yml configuration file which specifies all hyperparameters and training parameters. All configuration files required to replicate our experiments are collected in this [directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/configs/GPS). Please ensure that all relevant configuration files are copied to `/configs/GPS` directory in the `GPS repository` (i.e. the original implementation of GPS graph transformer). As we mentioned in our paper, to evaluate robustness of our approach we trained and evaluated GPS graph transformer networks using three different random seeds: 42,56,89.

For training use commands like: 

`python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest.yaml run_multiple_splits [0,1,2,3,4] seed 42`

(Note: replace the configuration file name, and seed number)

_<a name="part2-4">2.4. Inference with GPS graph transformer networks:</a>_

The inference can be similarly achieved. To do so, use commands like: 

`python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference.yaml run_multiple_splits [0,1,2,3,4] seed 42`

(Note: replace the configuration file name, and seed number)
