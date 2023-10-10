# GT-Remaining-CycleTime
GPS graph transformers can be used to predict remaining cyle time of business process instances in a two-step approach:

**<a name="part1">1. Data preparation:</a>**
Converting event logs into graph datasets. For more information, see: [conversion directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion). We already uploaded generated graph datasets in [conversion/transformation directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/transformation) in this repository. Therefore, this step can be skipped if you are not intreseted in conducting more experiments with feature engineering. In this case, generated graph dataset are directly used in the second step to train and evaluaate GPS graph transformers.

**<a name="part2">2. Training and evaluation of GPS graph transformers:</a>**

_<a name="part2-1">2.1. Original implementation of the GPS graph transformers, and setting up a python environment:</a>_

Our porposal is based on the original implementation of [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS). Therefore, the first step is to clone the original repository and follow the [instructions](https://github.com/rampasek/GraphGPS#python-environment-setup-with-conda) for setting up a Python environement with Conda.

_<a name="part2-2">2.2. Adjustments to the original implementation:</a>_

Once requirements of [2.1.](https://github.com/keyvan-amiri/GT-Remaining-CycleTime#part2-1) are met, the cloned version of the original implementation should be adjusted as per follows:
1. Replace `main.py` file in the root directory of the original implementation by [the one with same name](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/blob/main/main.py) in this repository. Here, the most important change is that a train mode called 'event-inference' is added to customize the inference step. 
