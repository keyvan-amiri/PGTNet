**Instructions:**

Name of the dataset must be the same one that is assigned to a variable called `dataset_name`  in the relevant configuration (.yml).


**Dataset Structure:**

For each dataset, three separate files are generated for the training, validation, and test sets. These files are formatted as graph dataset objects compatible with PyTorch Geometric library. Loading a graph dataset to train a GPS graph transformer is done using one single zip file including all these three parts. While our evaluation relies on cross-validation data split, we initially create separate graph dataset files for direct use in a holdout approach. Modifying data split approach can be easily done by using a variable called `split_mode` in the relevant  `.yml` file. 
