import os
import shutil

def copy_files_with_patterns(source_folders, target_folders, file_patterns):
    for source, target in zip(source_folders, target_folders):
        # Ensure source and target folders exist
        if not os.path.exists(source):
            print(f"Source folder '{source}' does not exist. Skipping.")
            continue
        if not os.path.exists(target):
            os.makedirs(target)

        # Get a list of files in the source folder that match the specified patterns
        files_to_copy = [file for file in os.listdir(source) if any(pattern in file for pattern in file_patterns)]

        # Copy each matching file to the target folder
        for file_name in files_to_copy:
            source_path = os.path.join(source, file_name)
            target_path = os.path.join(target, file_name)
            shutil.copy2(source_path, target_path)
            #print(f"Copied '{file_name}'.")

if __name__ == "__main__":
    # Get the path to the dir containing file transfer script (Root directory for PGTNet)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_path = os.path.join(script_dir, "scripts") #path to the scripts folder
    gps_directory = os.path.dirname(script_dir) #Root directory for GraphGPS
    loader_path = os.path.join(gps_directory, "graphgps", "loader")
    dataset_handler_path = os.path.join(loader_path, "dataset")
    encoder_path = os.path.join(gps_directory, "graphgps", "encoder")
    
    main_command_file = "main.py"
    main_command_source_path = os.path.join(scripts_path, main_command_file)
    main_command_destination_path = os.path.join(gps_directory, main_command_file)
    shutil.copy(main_command_source_path, main_command_destination_path)
    print(f"File copied from {main_command_source_path} to {main_command_destination_path}")
    
    master_loader_file = "master_loader.py"
    master_loader_source_path = os.path.join(scripts_path, master_loader_file)
    master_loader_destination_path = os.path.join(loader_path, master_loader_file)
    shutil.copy(master_loader_source_path, master_loader_destination_path)
    print(f"File copied from {master_loader_source_path} to {master_loader_destination_path}")
    
    event_handler_file = "GTeventlogHandler.py"
    event_handler_source_path = os.path.join(scripts_path, event_handler_file)
    event_handler_destination_path = os.path.join(dataset_handler_path, event_handler_file)
    shutil.copy(event_handler_source_path, event_handler_destination_path)
    print(f"File copied from {event_handler_source_path} to {event_handler_destination_path}")
    
    linear_edge_file = "linear_edge_encoder.py"
    linear_edge_source_path = os.path.join(scripts_path, linear_edge_file)
    linear_edge_destination_path = os.path.join(encoder_path, linear_edge_file)
    shutil.copy(linear_edge_source_path, linear_edge_destination_path)
    print(f"File copied from {linear_edge_source_path} to {linear_edge_destination_path}")
    
    two_layer_linear_edge_file = "two_layer_linear_edge_encoder.py"
    two_layer_linear_edge_source_path = os.path.join(scripts_path, two_layer_linear_edge_file)
    two_layer_linear_edge_destination_path = os.path.join(encoder_path, two_layer_linear_edge_file)
    shutil.copy(two_layer_linear_edge_source_path, two_layer_linear_edge_destination_path)
    print(f"File copied from {two_layer_linear_edge_source_path} to {two_layer_linear_edge_destination_path}")



    
"""
    # Construct the path to the "data" folder
    local_folder_path = os.path.join(script_dir, "..")
    kge_folder_path = os.path.join(local_folder_path, "..")
    data_folder_path = os.path.join(kge_folder_path, "data")
    preprocess_folder_path = os.path.join(data_folder_path, "preprocess")
    # define dataset names
    dataset_names = ["bpmai_lastrev_caise_onlyAfter", "bpmai_lastrev_caise_onlyAfter2",
                     "bpmai_lastrev_caise_inProcess", "bpmai_lastrev_caise_inProcess2",
                     "bpmai_lastrev_caise", "bpmai_lastrev_caise2"]
    # Specify the file patterns for copy
    file_patterns = [".txt", ".yaml"]
    
    # Transfer all dataset files 
    source_folders, target_folders = [], []
    for i in range (6):
        source_folders.append(os.path.join(bpmai_path, dataset_names[i]))
        target_folders.append(os.path.join(data_folder_path, dataset_names[i]))

    copy_files_with_patterns(source_folders, target_folders, file_patterns)
    
    # Transfer all configuration files 
    source_folders, target_folders = [], []
    for i in range (6):
        source_folders.append(os.path.join(configs_path, dataset_names[i]))
        target_folders.append(os.path.join(data_folder_path, dataset_names[i]))

    copy_files_with_patterns(source_folders, target_folders, file_patterns)
"""