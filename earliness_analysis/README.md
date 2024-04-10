Here, we present our implementation of DALSTM model.  

All data attributes that are used by DALSTM model are included `preprocessing_config.yaml` file. Note that only categorical attributes in event level can be used by DALSTM model. We used the same attributes that are used by PGTNet to have a fair comparison. 

In our pipeline, we only implemented preprocessing and holdout data split (training and inference) for seed 42. For cross-fold validation data split, we used three different random seeds to have fair comparison. If you want to use this code in other context or settings, do not remember to adjust the code.
