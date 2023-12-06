import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class EVENTBPIC15M1(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m1_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M1"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m1_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 395 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 219] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 1486 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))  
            
            
class EVENTBPIC15M2(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m2_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M2"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m2_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 405 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 142] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 1325,96 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M3(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m3_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M3"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m3_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 379 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 209] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 1512 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M4(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m4_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M4"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m4_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 353 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 141] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 926,9583 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M5(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m5_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M5"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m5_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 388 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 208] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 1343,96 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            

class EVENTBPIC12(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12_graph_raw.zip'  

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 35 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 77] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 137.221 (days)
                """
                
                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC12A(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12a_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12A"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12a_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 7 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 69] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 91.45 (days)
                """              

                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

            
class EVENTBPIC12C(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12c_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12C"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12c_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 22 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 77] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 91.455 (days)
                """   
                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC12CW(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12cw_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12CW"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12cw_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 5 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 68] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 91.04 (days)
                """ 
                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC12O(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12o_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12O"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12o_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 4 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 68] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 89.55 (days)
                """                 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            

class EVENTBPIC12W(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12w_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12W"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12w_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 18 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 68] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 137.221 (days)
                """                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            

class EVENTBPIC13C(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic13c_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC13C"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic13c_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 6 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 93] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 2254.85 (days)
                """                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC13I(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic13i_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC13I"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic13i_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 11 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 115] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 771.35 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC20D(InMemoryDataset):
   
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi20d_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC20D"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi20d_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 16 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 17] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 469.236 (days)
                """                 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC20I(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi20i_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC20I"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi20i_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 33 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 407] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 742 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTEnvPermit(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/envpermit_graph_raw.zip'   

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTEnvPermit"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'envpermit_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 25 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 68] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 275.84 (days)
                """                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTHelpDesk(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/helpdesk_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTHelpDesk"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'helpdesk_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 13 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 477] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 59,995 (days)
                """     
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTHospital(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/hospital_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTHospital"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'hospital_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 17 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 150] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 1035.42 (days)
                """                 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTSepsis(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/sepsis_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTSepsis"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'sepsis_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 15 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 250] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 422.32 (days)
                """                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTTrafficfines(InMemoryDataset):
    
    url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/trafficfines_graph_raw.zip'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTTrafficfines"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'trafficfines_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                """
                Each `graph` is a Pytorch Geometrioc Data object:
                    Shape of x : [num_nodes, 1] any integer from 0 to 10 represents one type of activity
                    class i.e. unique (activity identifier, lifecycle transition) tuples similar to
                    different types of atoms.
                    Shape of edge_attr : [num_edges, 267] the edge attributes represent weight of Directly
                    Followed realationships, temporal features, number of active cases in the process,
                    as well as data perspective in both case and event level.                     
                    Shape of edge_index : [2, num_edges] edge index determines topology of the graph. Our
                    main assumption is that process variants can be expressively represented by type of
                    nodes, and the topology of the graph (note that weight information is included in edge
                                                          attributes)
                    Shape of y : [1] a number between zero and one if normalized. You are more than
                    welcome to explore possibilities for learning without normalization. But, here:
                    normalizization coeficient = 4372 (days)
                """                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

# extra classes for ablation study
class EVENTBPIC13CABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/na5b8fvt4lwoy2euyaqna/bpic2013cablation_graph_raw.zip?rlkey=heoz0q77lw9qqwmnml3msvamz&dl=1'    
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC13CABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic2013cablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

            
class EVENTBPIC15M1ABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/yj2dekmdm1mdiyb0oxysz/bpic15m1ablation_graph_raw.zip?rlkey=3r3y7yf7w1ucpo5nkuhqcfrcn&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M1ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m1ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]           
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))  
            
            
class EVENTBPIC15M2ABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/wokzgkuu220vat7vquods/bpic15m2ablation_graph_raw.zip?rlkey=8gnokg68bjzqcpt6ljy89uvpf&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M2ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m2ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]             
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M3ABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/lze920t9p97f9bzgcdsmw/bpic15m3ablation_graph_raw.zip?rlkey=9j05n5g90oxc1ziea3o4vw2f0&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M3ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m3ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]            
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M4ABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/n8jx1a9td3c1g0grz0k6o/bpic15m4ablation_graph_raw.zip?rlkey=zg1x4dhzd6ir44lpgq73fim7c&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M4ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m4ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]             
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC15M5ABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/o5ril3re4u8k8hj5d6hjw/bpic15m5ablation_graph_raw.zip?rlkey=l17kp36dxeo4c3tg792inxhvv&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC15M5ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic15m5ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            

class EVENTBPIC12ABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/5uhs6359bqxl6l3u7o8ye/bpi12ablation_graph_raw.zip?rlkey=hsne9yz5yichuezbpyrct388n&dl=1'  

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12ABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12ablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC12AABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/heq1why1w7x8503ev9pda/bpi12aablation_graph_raw.zip?rlkey=ns87s5gtr26ca70h1y3aqi0ou&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12AABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12aablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]          
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

            
class EVENTBPIC12CABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/89uikm1dqw9epr9lg144j/bpi12cablation_graph_raw.zip?rlkey=b2rgal3wsqcs3crskeeng6xqq&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12CABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12cablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC12CWABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/w8xkr6cpqy386qjcr09x2/bpi12cwablation_graph_raw.zip?rlkey=cerx8yrb9y33g2gkt0lihjz4r&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12CWABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12cwablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC12OABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/zaen2l3o3oavkqqbshkfv/bpi12oablation_graph_raw.zip?rlkey=68ynlueg5d7lcl4sgoak0sarn&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12OABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12oablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]             
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            

class EVENTBPIC12WABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/rj1jf2k76fl6e7xg2xuty/bpi12wablation_graph_raw.zip?rlkey=s4btp7mfky62nmmo9p9ho35gu&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC12WABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi12wablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC13IABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/yyf54kr9g58cdspx5kfsz/bpic13iablation_graph_raw.zip?rlkey=n5istaipu722fqw8pn1sh5v55&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC13IABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpic13iablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]               
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTBPIC20DABLATION(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/a5e8deqixe200ns1y096q/bpi20dablation_graph_raw.zip?rlkey=814p7t70xgv3nk91e4jgeoqrt&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC20DABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi20dablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTBPIC20IABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/gcld1siqlsjwwonudwxhx/bpi20iablation_graph_raw.zip?rlkey=q97apbpwzu5ixsjf6w37jscd6&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTBPIC20IABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'bpi20iablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]               
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTEnvPermitABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/thrlxzhak56o5985ujr99/envpermitablation_graph_raw.zip?rlkey=3j6o95kfhjs1xpdjk0mdj5qx2&dl=1'   

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTEnvPermitABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'envpermitablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]                
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTHelpDeskABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/o4rpr7fypm1tbrv0xw9gx/helpdeskablation_graph_raw.zip?rlkey=3ci0yx9o8v258ib52ks2e56wj&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTHelpDeskABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'helpdeskablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]    
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                cid = graph.cid
                pl = graph.pl
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cid = cid, pl = pl)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTHospitalABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/1vwromqt0xkesxxldby9r/hospitalablation_graph_raw.zip?rlkey=grre5lyxhe4wx9nkmy21feh9i&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTHospitalABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'hospitalablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]               
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
class EVENTSepsisABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/muanfmbllyb5gnczzvfsg/sepsisablation_graph_raw.zip?rlkey=hulgh5gysfd7d6gjr33c3q0a7&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTSepsisABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'sepsisablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]               
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            
            
class EVENTTrafficfinesABLATION(InMemoryDataset):
    
    url = 'https://www.dropbox.com/scl/fi/nduy4ya31zbp9eu0qdni0/trafficfinesablation_graph_raw.zip?rlkey=87s8ftmeag7vedxpsqa25zsmi&dl=1'    

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "EVENTTrafficfinesABLATION"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'trafficfinesablation_graph_raw'), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]                  
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                edge_weight = graph.edge_weight
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                            edge_weight = edge_weight)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))