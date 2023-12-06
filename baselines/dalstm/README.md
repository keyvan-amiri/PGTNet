This directory includes all adjustments required for implementation of DALSTM baseline. 

In order to extend this baselines to more event logs, `attributes.yaml` should be adjusted. This file includes name of all data attributes that are used by DALSTM. For instance, to extend DALSTM implementation to the BPIC15-1 event log we have to add the followings to the .yaml file:

"BPIC15_1" : ['org:resource', 'monitoringResource']

In order to replicate our experiments with cross-validation data split the implementation of **pmdlcompararator** repository should be adjusted based on the content of this  [**folder**](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/dalstm/crossvalidation), while for holdout data split contents of this [**folder**](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/dalstm/holdout) are relevant.
