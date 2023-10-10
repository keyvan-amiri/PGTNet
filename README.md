# GT-Remaining-CycleTime
GPS graph transformers can be used to predict remaining cyle time of business process instances in a two-step approach:

**1. Data preparation:**
Converting event logs into graph datasets. For more information, see: [conversion directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion). We already uploaded generated graph datasets in [conversion/transformation directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/transformation) in this repository. Therefore, this step can be skipped if you are not intreseted in conducting more experiments with feature engineering. In this case, generated graph dataset are directly used in the second step to train and evaluaate GPS graph transformers.

**2. Training and evaluation of GPS graph transformers:**
