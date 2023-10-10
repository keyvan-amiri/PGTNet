import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

class EVENTBPIC15M1(InMemoryDataset):
   
    url = 'https://www.dropbox.com/scl/fi/t8721bgrbxvjag81rq3wu/bpic15m1_graph_raw.zip?rlkey=xr731a8sv8p7ox2sit8aiwsfv&dl=1'    

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
   
    url = 'https://www.dropbox.com/scl/fi/6l87mr73cp123fq2e0so8/bpic15m2_graph_raw.zip?rlkey=1zn3mmqs4glp4d024o8vul39m&dl=1'    

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
   
    url = 'https://www.dropbox.com/scl/fi/utueh0a8fndhw38gno3j0/bpic15m3_graph_raw.zip?rlkey=zyu0zi4otr1xcr2y1notr0b1m&dl=1'    

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
   
    url = 'https://www.dropbox.com/scl/fi/lmvoe8klh8144n7q5mjiv/bpic15m4_graph_raw.zip?rlkey=id6ker4yoo5wrrxyi66afzrzp&dl=1'    

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
   
    url = 'https://www.dropbox.com/scl/fi/0xspwb9dqcy34ds7qsdwl/bpic15m5_graph_raw.zip?rlkey=blwvo52pas8mzxd8xz28h4wd2&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/52xl2kpsxv7oo3v1id4y4/bpi12_graph_raw.zip?rlkey=hpw16ih899wur5w76qhnpeluo&dl=1'    

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
    
    url = 'https://www.dropbox.com/s/yobrnl8r9fhj7hb/bpi12a_graph_raw.zip?dl=1'    

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
    
    url = 'https://www.dropbox.com/s/h39f57jrj27f2lr/bpi12c_graph_raw.zip?dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/l4w5zkqot66k29ezctre1/bpi12cw_graph_raw.zip?rlkey=ttgow4pb92byq5lhxtrx89ezo&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/3shh4b08unj6k1dzl0iz7/bpi12o_graph_raw.zip?rlkey=82i4wmet4ebfv7103n2whrzji&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/s8axejhyoq6vw231k0siq/bpi12w_graph_raw.zip?rlkey=0yryk23g9vix299c1jtq10afz&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/dkts7lia4qsffaziifuu3/bpic13c_graph_raw.zip?rlkey=717lxd6b7pd20i98d6fdmuzje&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/4wb6jklq3pxfsqi4g7izw/bpic13i_graph_raw.zip?rlkey=hb5tfy102cxqtkyd9f37rb4wr&dl=1'    

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
   
    url = 'https://www.dropbox.com/scl/fi/53bvoactg7oz9mvfzx9d2/bpi20d_graph_raw.zip?rlkey=ghbmhregosejstsvy8b0l13wd&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/6697fuek1zeoay78zdboi/bpi20i_graph_raw.zip?rlkey=l962dw3sflweyher8kvg2eyjy&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/hgcb9lungzo2zoqdnq77a/envpermit_graph_raw.zip?rlkey=4h7ebsbd58a6hah7lo61pv1o1&dl=1'   

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
    
    url = 'https://www.dropbox.com/scl/fi/jtsc9icb97kdnttqafdy5/helpdesk_graph_raw.zip?rlkey=nu4vtb0v9nf2a6czu78d74t4n&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/odc2t7iun3m9kydzp27gw/hospital_graph_raw.zip?rlkey=afefqda0674rgbup1bdfjsozr&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/1nbohmnkqwlxbsh7mwtd9/sepsis_graph_raw.zip?rlkey=mny968wrdgsykjf1myvffg1zb&dl=1'    

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
    
    url = 'https://www.dropbox.com/scl/fi/uxc920zjf7h00qmukbniz/trafficfines_graph_raw.zip?rlkey=lhspdniqui78whgrv1v6cisof&dl=1'    

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