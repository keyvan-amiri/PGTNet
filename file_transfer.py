import os
import shutil

if __name__ == "__main__":
    # Get important paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) #Root directory for PGTNet
    scripts_path = os.path.join(script_dir, "scripts") #path to the scripts folder
    train_conf_path = os.path.join(script_dir, "training_configs") #path to the training_configs folder
    eval_conf_path = os.path.join(script_dir, "evaluation_configs") #path to the evaluation_configs folder
    gps_directory = os.path.dirname(script_dir) #Root directory for GraphGPS
    loader_path = os.path.join(gps_directory, "graphgps", "loader")
    dataset_handler_path = os.path.join(loader_path, "dataset")
    encoder_path = os.path.join(gps_directory, "graphgps", "encoder")
    gps_config_path = os.path.join(gps_directory, "configs", "GPS")  #path to the GPS configs  
    
    main_command_file = "main.py"
    main_command_source_path = os.path.join(scripts_path, main_command_file)
    main_command_destination_path = os.path.join(gps_directory, main_command_file)
    shutil.copy(main_command_source_path, main_command_destination_path)
    #print(f"File copied from {main_command_source_path} to {main_command_destination_path}")
    
    master_loader_file = "master_loader.py"
    master_loader_source_path = os.path.join(scripts_path, master_loader_file)
    master_loader_destination_path = os.path.join(loader_path, master_loader_file)
    shutil.copy(master_loader_source_path, master_loader_destination_path)
    #print(f"File copied from {master_loader_source_path} to {master_loader_destination_path}")
    
    event_handler_file = "GTeventlogHandler.py"
    event_handler_source_path = os.path.join(scripts_path, event_handler_file)
    event_handler_destination_path = os.path.join(dataset_handler_path, event_handler_file)
    shutil.copy(event_handler_source_path, event_handler_destination_path)
    #print(f"File copied from {event_handler_source_path} to {event_handler_destination_path}")
    
    linear_edge_file = "linear_edge_encoder.py"
    linear_edge_source_path = os.path.join(scripts_path, linear_edge_file)
    linear_edge_destination_path = os.path.join(encoder_path, linear_edge_file)
    shutil.copy(linear_edge_source_path, linear_edge_destination_path)
    #print(f"File copied from {linear_edge_source_path} to {linear_edge_destination_path}")
    
    two_layer_linear_edge_file = "two_layer_linear_edge_encoder.py"
    two_layer_linear_edge_source_path = os.path.join(scripts_path, two_layer_linear_edge_file)
    two_layer_linear_edge_destination_path = os.path.join(encoder_path, two_layer_linear_edge_file)
    shutil.copy(two_layer_linear_edge_source_path, two_layer_linear_edge_destination_path)
    #print(f"File copied from {two_layer_linear_edge_source_path} to {two_layer_linear_edge_destination_path}")

    # Move all training configurations    
    files = [f for f in os.listdir(train_conf_path) if os.path.isfile(os.path.join(train_conf_path, f))]
    for file in files:
        source_path = os.path.join(train_conf_path, file)
        destination_path = os.path.join(gps_config_path, file)
        shutil.copy(source_path, destination_path) 
    #print(f"All configurations from {train_conf_path} to {gps_config_path}")
    # Move all evaluation configurations    
    files = [f for f in os.listdir(eval_conf_path) if os.path.isfile(os.path.join(eval_conf_path, f))]
    for file in files:
        source_path = os.path.join(eval_conf_path, file)
        destination_path = os.path.join(gps_config_path, file)
        shutil.copy(source_path, destination_path) 
    #print(f"All configurations from {eval_conf_path} to {gps_config_path}")
