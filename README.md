# PGTNet: A Process Graph Transformer Network for Remaining Time Prediction of Business Process Instances
This is the supplementary githob repository of the paper: "PGTNet: A Process Graph Transformer Network for Remaining Time Prediction of Business Process Instances".

Our approach consists of a data transformation from an event log to a graph dataset, and training a neural network based on the [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS) recipe. 
<p align="center">
  <img src="https://github.com/keyvan-amiri/PGTNet/blob/main/PGTNet-Architecture.png">
</p>

**<a name="part1">1. Set up a Python environement to work with GPS Graph Transformers:</a>**

GPS Graph Transformers recipe is implemented based on the PyTorch machine learning framework, and it utlizes [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) (PyTorch Geometric) library. In order to be able to work with GPS Graph Transformers, you need to set up a Python environement with Conda as suggested [here](https://github.com/rampasek/GraphGPS#python-environment-setup-with-conda). To set up such an environement:
```
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

pip install pm4py
#pip install PyYAML #Requirement already satisfied
#pip install numpy  #Requirement already satisfied
#pip install scikit-learn #Requirement already satisfied

conda clean --all
```
Note that, we also included pip install for [pm4py](https://pm4py.fit.fraunhofer.de/) library to facilitate working with event log data. 

**<a name="part2">2. Clone repositories and download event logs:</a>**

Once you setup a conda environement, clone the [GPS Graph Transformer repository](https://github.com/rampasek/GraphGPS) using the following command:
```
git clone https://github.com/rampasek/GraphGPS
```
This repository is called **GPS repository** in the remaining of this README file. Now, Navigate to the root directory of **GPS repository**, and clone the current repository (i.e., the **PGTNet repository**). By doing so, the **PGTNet repository** will be placed in the root directory of **GPS repository** meaning that the latter is the parent directory of the former.
```
cd GraphGPS
git clone https://github.com/keyvan-amiri/PGTNet
```
Now, we are ready to download all event logs that are used in our experiments. In priniciple, downloading event logs and converting them to graph datasets are not mandatory steps for training PGTNet because we already uploaded the resultant graph datasets [here](https://github.com/keyvan-amiri/PGTNet/tree/main/transformation). In case you want to start with [training](https://github.com/keyvan-amiri/PGTNet#part4) a PGTNet, you can skip this step and the next on. In this case, generated graph dataset are automatically downloaded and will be used to train and evaluate PGTNet for remaining time prediction.

To download all event logs, navigate to the root directory of **PGTNet repository** and run `data-acquisition.py` script:
```
cd PGTNet
python data-acquisition.py
```
All event logs utilized in our experiments are publicly available at [the 4TU Research Data repository](https://data.4tu.nl/categories/13500?categories=13503). The **data-acquisition.py** script download all event logs, and convert them into .xes format. It also generates additional event logs (BPIC12C, BPIC12W, BPIC12CW, BPIC12A, BPIC12O) from BPIC12 event log.  

**<a name="part3">3. Converting an event log into a graph dataset:</a>**

In order to convert an event log into its corresponding graph dataset, you need to run the same python script with specific arguments:
```
python GTconvertor.py conversion_configs bpic15m1.yaml --overwrite true
```
The first argument (i.e., conversion_configs) is a name of directory in which all required configuration files are located. The second argument (i.e., bpic15m1.yaml) is the name of configuration file that is used for conversion. We will discuss this argument with more details in the followings. The last argument called overwrite is a Boolean variable which provides some flexibility. If it is set to false, and you have already converted the event log into its corresponding graph dataset the script simply skip repeating the task. 

[_Conversion Configuration Files:_](https://github.com/keyvan-amiri/PGTNet/tree/main/conversion_configs)
Each conversion configuration file defines parameters used for converting the event log into its corresponding graph dataset:
| Parameter name | Parameter description |
|----------|----------|
| raw_dataset  | Name of the raw dataset (i.e., an event log in xes format).| 
| event_attributes  | Name of the categorical attributes in event-level that are included in conversion process. **1**| 
| event_num_att  | Name of the numerical attributes in event-level that are included in conversion process. **1**| 
| case_attributes  | Name of the categorical attributes in case-level that are included in conversion process. **1**| 
| case_num_att  |  Name of the numerical attributes in case-level that are included in conversion process. **1**|
| train_val_test_ratio  |  Training, validation, and test data split ratio. By default, we use a 0.64-0.16-0.20 data split ratio. This means that we sort all traces based on the timestamps of their first events, and then use the first 64% for training set, the next 16% for validation set and the last 20% for test set. **2**|
| target_normalization  | A boolean attribute (default: true) which specifies the normalization of target attribute. If set to true, the target attribute is normalized based on the duration of the longest case, ensuring values fall within the range of zero to one. **3**|
<!-- This is not remaining of the table. -->
**1.** The implementation provides the opportunity to experiment with different combinations for these variables. Therefore, it is easy to conduct ablation studies or investigate contribution of different attributes to the accuracy of predictions.
**2.** This is equivalent to holdout data split. Later, we will discuss how we can use cross-fold validation data split using training configuration files.
**3.** This normalization proved to be helpful because the target attribuite often has a highly skewed distribution.

**Graph dataset structure:** The resultant graph dataset will be saved in a seperate folder which is located in the **datasets** folder in the root directory for **GPS repository**. Each graph dataset is a [PyG data object](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) and represents a set of event prefixes (each attributed directed graph corresponds to an event prefix: an unfinished business process instance). For each graph dataset, three separate files are generated for the training, validation, and test sets. These files are formatted as graph dataset objects compatible with PyTorch Geometric library. While our evaluation relies on cross-validation data split, we initially create separate graph dataset files for direct use in the holdout approach. Modifying data split approach can be easily done by using a variable called **split_mode** in the relevant training configuration file. 

Note that, Running the `GTconvertor.py` script produces several additional output files, including:
1. Encoders: One-hot encoders for both case-level and event-level attributes, implemented using scikit-learn.
2. Activity Classes Dictionary: A dictionary that defines activity classes.
3. Filtered Cases: A list of case IDs for cases that do not have at least three events.
4. Histogram: A PNG file that visualizes the distribution of target attribute values.

All additional ouputs are saved in a separate folder called **transformation** in the root directory of **PGTNet repository**. 

**Note:** We provide additional text files describing general statistics for different graph datasets. See: [General statistics for graph datasets](https://github.com/keyvan-amiri/PGTNet/tree/main/graph_dataset_statistics).

**<a name="part4">4. Training a PGTNet for remaining time prediction:</a>**

To train and evaluate PGTNet, we employ the implementation of [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS). However, in order to use it for remaining time prediction of business process instances, you need to adjust some part of the original implementation. This can be achieved by running the following command:
```
python file_transfer.py
```
This script copies 5 important python scripts which take care of all necessary adjustments to the original implementation of GPS Graph Transformer recipe:

a. In **main.py** , the most important change is that a train mode called 'event-inference' is added to customize the inference step.

b. In **master_loader.py**, the most important change is that several new dataset classes are added to handle graph representation of event logs.

c. The python script **GTeventlogHandler.py** includes multiple **InMemoryDataset** Pytorch Geometric classes. We created one seperate class for each event log.

d. The python scripts **linear_edge_encoder.py** and **two_layer_linear_edge_encoder.py** are specifically designed for edge embedding in the remaining cycle time prediction problem.

Once abovementioned adjustments are done, training PGTNet is straightforward. Training is done using the relevant .yml configuration file which specifies all hyperparameters and training parameters. All configuration files required to train PGTNet based on the event logs used in our experiments are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/training_configs). The **file_transfer.py** script also copy all required configuration files for training and evaluation of PGTNet to the relevant folder in **GPS repository**.

For training PGTNet, you need to navigate to the root directory of **GPS repository** and run **main.py** script:
```
cd ..
python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest.yaml run_multiple_splits [0,1,2,3,4] seed 42
```
As we mentioned in our paper, to evaluate robustness of our approach we trained and evaluated PGTNet using three different random seeds. These random seeds are 42, 56, 89. Each time you want to train PGTNet for specific event log and specific seed number, you should adjust the **training configuration file** name, and the seed number in this command.

The [**training configuration files**](https://github.com/keyvan-amiri/PGTNet/tree/main/training_configs) include all required training hyperparameters. Following table briefly discusses the most important parameters:

| Parameter name | Parameter description |
|----------|----------|
| out_dir  | Name of the directory in which the results will be saved (e.g., results)| 
| metric_best | The metric that is used for results. In our case, it is always **"mae"** (Mean Absolute Error). There is another parameter called "metric_agg" which determines whether the metric should be maximized or minimized (in our case it is always set to "argmin".)| 
| dataset.format | Name of the PyG data object class that is used (e.g., PyG-EVENTBPIC15M1)| 
| dataset.task | Specifies the task level. In our case, it is always set to **"graph"** since we always have a graph-level prediction task at hand.| 
| dataset.task_type | Specifies the task type. In our case, it is always set to **"regression"**.| 
| dataset.split_mode | while **"cv-kfold-5"** specifies cross-fold validation data split, **"standard"** can be used for holdout data split.| 
| node_encoder_name | Specifies the encoding that will be employed for nodes. For instance, in **"TypeDictNode+LapPE+RWSE"**, "TypeDictNode" refers to embedding layer, and "LapPE+RWSE" refers to the type of PE/SEs that are used. There is another parameter called **node_encoder_num_types** which should be set the number of activity classes in the event log. For instance, node_encoder_num_types: 396 for the BPIC15-1 event log.| 
| edge_encoder_name | Specifies the encoding that will be employed for edges. For instance, "TwoLayerLinearEdge" refers to two linear layers.| 
| PE/SE parameters | Depending of type of PE/SEs that are used, all relevant hyperparameter can be defined. For instance if "LapPE+RWSE" is used, hyperparameters can be defined using "posenc_LapPE" and "posenc_RWSE". These hyperparameters include a wide range of options for instance the size of PE can be defined using "dim_pe", and the model that is used for processing it can be defined using "model" (for instance, model: DeepSet).|
| train | Specify the most important training hyperparameters including the training mode (i.e., **train.mode**) and batch size (i.e., **train.batch_size**). We always use the **custom** mode for training. |
| model | Specifies the most important global design options. For instance, **model.type** defines type of the model and in our case is always a **GPSModel**. The **model.loss_fun** defines the loss fucntion which in our case is always **l1** (the L1 loss function is equivalent to Mean Absolute Error). The **model.graph_pooling** specifies type of graph pooling and for instance can be set to "mean".|
| gt | Specifies the most important design options with respect to Graph Transformer that will be employed. For instance, **gt.layer_type** defines type of MPNN and Transformer blocks within each GPS layer, and in our case is always set to **GINE+Transformer**. The **gt.layers** and **gt.n_heads** define the number of GPS layers and number of heads in each layer. The **gt.dim_hidden** defines the hidden dimentsion that is used for both node and edge features. Note that, this size also include PE/SEs that are incorporated into node and edge features. The **gt.dropout** and **gt.attn_dropout** define the dropout value for MPNN and Transformer blocks, respectively.|
| optim | Specifies the most important design options with respect to the optimizer. For instance, **optim.optimizer** specifies the optimizer type and in our case is always set to **adamW**. The **optim.base_lr** and **optim.weight_decay** define base learning rate and weight decay, respectively. The **optim.max_epoch** specifies number of training epochs, while **optim.scheduler** and **optim.num_warmup_epochs** specify type of schedule (in our case always **cosine_with_warmup**) and number of warmup epochs. |
<!-- This is not remaining of the table. -->

Training results are saved in a seperate folder which is located in the **results** folder in the root directory of **GPS repository**. Name of this folder is always equivalent to the name of the configuration file that is used for training. For instance running the previous command produces this folder: **bpic2015m1-GPS+LapPE+RWSE-ckptbest**

This folder contains the best models (i.e., checkpoints) for each of 5 different folds. The checkpoint files can be used for inference with PGTNet  
based on the validation error  from training step

**<a name="part5">5. Inference with PGTNet:</a>**

The inference (i.e., get prediction of PGTNet for all examples in the test set) can be done similar to the training step. To do so, run commands like: 
```
python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference.yaml run_multiple_splits [0,1,2,3,4] seed 42
```
All **inference configuration files** that are used in our experiments are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/evaluation_configs).

In principle, the inference configuration files are similar to the training configuration files. The most important difference is that, the **"train.mode"** parameter is set to **"event-inference"** instead of "custom". The inference configuration files additionally include another parameter called **"pretrained.dir"** by which we specify the folder that contais training results. For instance, it can be something like this:
```
pretrained:
  dir: /home/kamiriel/GraphGPS/results/bpic2015m1-GPS+LapPE+RWSE-ckptbest #the location of the training results on your system
```

Running the inference script results in one dataframe (.csv) for each fold. Each row in this dataframe represent a test example for which the number of nodes, the number of edges, real remaining time and predicted remaining time are provided thorugh these columns: "num_node","num_edge","real_cycle_time","predicted_cycle_time". These files still need to be processed in order to: 1) provide the aggregated results over 5 folds, 2) match the rows to event prefixes, and 3) provide errors in days rather then normalized numbers in "real_cycle_time","predicted_cycle_time". This can be achived by navigating to the root directory of **PGTNet repository** and running the following script:
```
cd PGTNet
python ResultHandler.py --dataset_name 2015m1 --seed_number 42 --inference_config 'bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference'
```
The aggregated dataframe will be saved in a folder called **PGTNet results** in the root directory of **PGTNet repository**, and can be used for further analysis with respect to accuracy and earliness of predictions.

**<a name="part6">6. Miscellaneous:</a>**

**Earliness of PGTNet's predictions:**

We are interested in models that not only have smaller MAE but also can make accurate predictions earlier, allowing more time for corrective actions. We used the method proposed in [Predictive Business Process Monitoring with LSTM Neural Networks](https://link.springer.com/chapter/10.1007/978-3-319-59536-8_30), which evaluates MAE across different event prefix lengths. In our paper, we have provided the predcition earliness analysis (i.e., MAE trends at different prefix lengths) only for BPIC15-4, Sepsis, Helpdesk, and BPIC12A event logs. Similar analysis for other event logs used in our experiments can be found [here](https://github.com/keyvan-amiri/PGTNet/tree/main/earliness_analysis).

**Ablation study:**

As it is discussed in our paper, we conducted an ablation study for which we trained a minimal PGTNet model, relying solely on edge weights (i.e., control-flow) and temporal features, thus omitting data attributes from consideration. To replicate our ablation study, you need to adjust the conversion script and use different configuration files which you can find [here](https://github.com/keyvan-amiri/PGTNet/tree/main/ablation_study). For the quantitative analysis of the contribution of PGTNet's architecture and the contribution of incorporating additional features to the remarkabel performance of PGTNet see this [plot](https://github.com/keyvan-amiri/PGTNet/blob/main/ablation_study/ablation_plot.pdf). 

**PGTNet's results for holdout data split:**

While we chose a 5-fold cross-validation strategy (CV=5) in our experiments, we also report the [results](https://github.com/keyvan-amiri/PGTNet/blob/main/holdout_results/README.md) obtained using holdout data splits for the sake of completeness. Note that, we used different training and configuration files for holdout data split which are not part of this repository.

**Implementation of the baselines:**
As it is discussed in our paper, we compare our approach against three others:
1. DUMMY : A simple baseline that predicts the average remaining time of all training prefixes with the same length k as a given prefix.
2. [DALSTM](https://ieeexplore.ieee.org/abstract/document/8285184): An LSTM-based approach that was recently shown to have superior results among LSTMs used for remaining time prediction. To implement this baseline, we used the [**pmdlcompararator**](https://gitlab.citius.usc.es/efren.rama/pmdlcompararator) github repository of a recently published [benchamrk](https://ieeexplore.ieee.org/abstract/document/9667311). DALSTM is implemented for both cross-validation and holdout data splits in **pmdlcompararator** repository. We extended its implemetation to some event logs that were not part of the benchmark by adjusting the relevant .yaml [file](https://github.com/keyvan-amiri/PGTNet/blob/main/baselines/dalstm/attributes.yaml). Additionally, the performance evaluation of DALSTM in the benchmark differs from our evaluation because we have not included predictions for prefixes of length 1 (remember that our graph representation of event prefixes requires at least two events). Therefore, we have adjusted some parts of **pmdlcompararator** repository as explained [here](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/dalstm). 
3. [ProcessTransformer](https://arxiv.org/abs/2104.00721): A transformer-based approach designed to overcome LSTM’s limitations that generally outperforms DALSTM. To implement this baseline, we used [**ProcessTransformer**](https://github.com/Zaharah/processtransformer) github repository. **ProcessTransformer** repository does not include the implementation for cross-validation data split. Furthermore, the performance evaluation of ProcessTransformer is conducted differently. Predictions of event prefixes of length 1, and length n (where n is the number of events in the trace) are not included in our performance evaluation. More importantly, the metric that is reported in ProcessTransformer paper is not MAE (mean absolute error) as authers computed the average of errors for different prefix lengths while they did not account for different frequencies of different prefix lengths in the test set. However, MAE should reflect frequencies and can be considered as the weighted average of errors for different prefix lengths. In order to have a fair comparison, we have adjusted some parts of **ProcessTransformer** repository as explained [here](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/processtransformer).
