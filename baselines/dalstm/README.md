Here, we present our implementation of DALSTM model.  

In order to extend this baselines to more event logs, `attributes.yaml` should be adjusted. This file includes name of all data attributes that are used by DALSTM. For instance, to extend DALSTM implementation to the BPIC15-1 event log we have to add the followings to the .yaml file:

"BPIC15_1" : ['org:resource', 'monitoringResource']


