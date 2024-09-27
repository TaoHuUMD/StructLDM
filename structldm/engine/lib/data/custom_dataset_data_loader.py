import torch.utils.data
from .base_data_loader import BaseDataLoader

import importlib
from munch import *

def CreateDataset(opt, phase):
    dataset = None

    dataset_name = None
    if isinstance(opt.dataset, dict) or isinstance(opt.dataset, Munch):
        dataset_name = opt.dataset.dataset_name
    else:
        dataset_name = opt.dataset

    print("load dataset ", dataset_name)
    
    proj_dir = opt.project_directory

    dataset_name_list = ["deepfashion", "zju", "mpi", "aist", "mpi_free"]
    data_dict = {}
    for s in dataset_name_list:
        data_dict.update({s: f'{proj_dir}.data.dataset_{s}'})


    if dataset_name in dataset_name_list:
        #from data_dict[opt.dataset] import 
        dataset = importlib.import_module(data_dict[dataset_name]).Dataset()
        dataset.initialize(opt, phase, opt.multi_datasets[0])
        print("dataset [%s] was created" % (dataset_name))
        return dataset
    else:
        raise NotImplementedError()
    
    # if opt.df:
    #     dataset = Dataset()
    #     dataset.initialize(opt, phase, opt.multi_datasets[0])
    #     print("deep fashion multi dataset was created")
    # elif opt.motion_mode:
    #     dataset = Dataset()
    #     dataset.initialize(opt, phase, opt.multi_datasets[0])
    #     print("motion dataset was created")
    # elif opt.multi_id: # and opt.smpld_nr
    #     all=[]
    #     for i in range(len(opt.multi_datasets)):            
    #         dataset = Dataset()
    #         #id = opt.dataset[opt.multi_datasets[i].resname]            
    #         dataset.initialize(opt, phase, opt.multi_datasets[i])
    #         all.append(dataset) 
    #     dataset = torch.utils.data.ConcatDataset(all)
    #     print("multi dataset was created")
    # else:
    #     dataset = Dataset()

    #     print("dataset [%s] was created" % (dataset.name()))
    #     dataset.initialize(opt, phase, opt.multi_datasets[0])
    
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, phase="train"):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, phase)

        batch_size = opt.batchSize
        nthreads = int(opt.nThreads)
        if phase == "evaluate":
            nthreads = 0
            batch_size = 1
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = batch_size,
            shuffle=not opt.serial_batches,
            #sampler = self.dataset.get_train_sampler() if phase == "train" else None,
            num_workers = int(nthreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
        #return min(len(self.dataset), self.opt.max_dataset_size)



class DistributedCustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'DistributedCustomDatasetDataLoader'

    # def __init__(self, opt, dataset, sampler, phase="train"):
    #     super().__init__()
    #     return self.initialize(opt, dataset, sampler, phase)

    def initialize(self, opt, dataset, sampler, phase="train"):

        assert phase=="train"
        BaseDataLoader.initialize(self, opt)
        #self.dataset = CreateDataset(opt, phase)
        
        self.dataset = dataset

        nthreads = int(opt.nThreads)
   
        print("distri ", phase, opt.batchSize)

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size= opt.batchSize,
            shuffle = True if sampler is None else False,
            sampler = sampler,
            num_workers = int(nthreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):        
        return len(self.dataset)
        #return min(len(self.dataset), self.opt.max_dataset_size)
