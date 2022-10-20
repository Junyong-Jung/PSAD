from argparse import ArgumentParser
from psad_dataset import PSAD_single_dataset, PSAD_multi_dataset
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks
import random
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
parser = ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--category',type=str, required=True)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("dir Create Done!")
        else:
            print("directory already exist!")
    except OSError:
        print ('Error: Failed to Create directory. ' +  directory)
        
def main():
    args = parser.parse_args()
    config = get_config('configs/config.yaml')
    config['normal_class'] = args.category
    vgg, model = get_networks(config, load_checkpoint=True)


    # Localization test
    if config['localization_test']:
        
        not_implemented_localization = ['multi_late_mean_pooling', 'multi_late_max_pooling', 'multi_late_mean_max_pooling']
        if config['version'] in not_implemented_localization:
            print(config['version']+'_localization test is not implemented')
            return None
        
        createFolder(os.path.join('heatmap_outputs',args.category))
        test_dataloader, ground_truth = load_localization_data(config)
        roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader,ground_truth=ground_truth, config=config)
        last_checkpoint = config['last_checkpoint']
        print("Localization RocAUC after {} epoch:".format(last_checkpoint), roc_auc)

    # Detection test
    else:
        normal_class = config['normal_class']
        test_data_path = os.path.join(config['dataset_path'],normal_class,'test')
        test_data_path = os.path.expanduser(test_data_path)
        defect_class_list = os.listdir(test_data_path)
        defect_class_list.remove('good')
        
        def run(defect_class):
            test_dataloader = defect_class_dataset(config, test_data_path, defect_class)
            roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
             
            last_checkpoint = config['last_checkpoint']
            print("good_{} RocAUC after {} epoch:".format(defect_class,last_checkpoint), roc_auc)
        
        
        run(defect_class='all')
        # for defect_class in defect_class_list:
        #     run(defect_class)


def defect_class_dataset(config, test_data_path, defect_class):
    mvtec_img_size = config['mvtec_img_size']
    batch_size = config['batch_size']
    
    orig_transform = transforms.Compose([
        transforms.Resize([mvtec_img_size, mvtec_img_size]),
        transforms.ToTensor()
    ])
    
    if 'single' in config['version']:
        test_set = PSAD_single_dataset(root=test_data_path, transform=orig_transform)
    elif 'multi' in config['version']:
        test_set = PSAD_multi_dataset(root=test_data_path, transform=orig_transform)

    if defect_class != 'all':
        if 'single' in config['version']:
            test_set.samples = [i for i in test_set.samples if (defect_class in i[0]) or ('good' in i[0])]
        elif 'multi' in config['version']:
            test_set.samples = [i for i in test_set.samples if (defect_class in i[0][0]) or ('good' in i[0][0])]

        
    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    return test_dataloader


if __name__ == '__main__':
    random.seed(100)
    main()

