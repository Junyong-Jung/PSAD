import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, make_dataset
from . import plus_variable
import random
import torch

class PSAD_single_dataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        
        
        self.samples = [i for i in self.samples if os.path.basename(i[0]) in plus_variable.file_list]
        def PSAD_target_transform(target):
            
            if target == self.class_to_idx["good"]:
                target = 0
            else:
                target = 1
            
            return target
                
        self.target_transform = PSAD_target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target        
    
class PSAD_multi_dataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        
        
        self.samples = [i for i in self.samples if os.path.basename(i[0]) in plus_variable.file_list]
        unit = plus_variable.num_imgs
        self.samples = [self.samples[i : i + unit] for i in range(0,len(self.samples),unit)]
        
        def jjy_target_transform(target):
            
            if target == self.class_to_idx["good"]:
                target = 0
            else:
                target = 1
            
            return target
                
        self.target_transform = jjy_target_transform
        
    def __getitem__(self, index):
        
        
        samples_list = self.samples[index]
        channel_concat_list = []
        random.shuffle(samples_list)
        for path, target in samples_list:
            sample = self.loader(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            channel_concat_list.append(sample)
            
        samples = torch.cat(channel_concat_list, dim=0)
        
        return samples, target        