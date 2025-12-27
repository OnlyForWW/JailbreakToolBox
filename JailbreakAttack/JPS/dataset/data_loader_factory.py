from .advbench_subset_loader import advbench_Subset_Loader

class DataLoaderFactory:
    @staticmethod
    def get_loader(config):
        dataset_name = config['dataset_name']
        if dataset_name == 'advbench_subset':
            return advbench_Subset_Loader(config)
        else:
            raise ValueError(f"未知的数据集类型: {dataset_name}")