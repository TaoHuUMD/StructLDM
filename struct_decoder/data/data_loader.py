
def CreateDataLoader(opt, phase="train"):
    from .custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, phase)
    return data_loader

def CreateDataLoaderDistributed(opt, dataset, sampler, phase="train"):
    from .custom_dataset_data_loader import DistributedCustomDatasetDataLoader
    data_loader = DistributedCustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, dataset, sampler, phase)
    return data_loader

def CreateDataset(opt, phase):
    from .custom_dataset_data_loader import CreateDataset as CreateDataset_
    return CreateDataset_(opt, phase)