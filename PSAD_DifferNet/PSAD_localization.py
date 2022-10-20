'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''
import argparse
from configparser import Interpolation
from turtle import Turtle
import config as c
from model import save_model, save_weights, load_model
import plus_variable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from skimage.segmentation import mark_boundaries
from skimage import morphology
from tqdm import tqdm

from localization import export_gradient_maps
from utils import *

from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os
from scipy.ndimage import rotate, gaussian_filter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.transforms import InterpolationMode
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

GRADIENT_MAP_DIR = 'PSADresult/grad_result'

def save_imgs(inputs, grad, np_gt, threshold, path_list, cnt):
    export_dir = os.path.join(GRADIENT_MAP_DIR, c.modelname)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for g in range(grad.shape[0]):
        unnormed_grad = grad[g]
        orig_image = inputs[g]
        np_gt_i = np_gt[g]
        path = path_list[g]
        plot_fig(orig_image, unnormed_grad, np_gt_i, threshold, export_dir, path, cnt)
        cnt += 1
    return cnt



def plot_fig(orig_image, score, np_gt, threshold, export_dir, path:str, cnt):
    mask = score.copy()
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    
    mask *= 255
    mask = mask.astype(np.uint8)
    
    heat_map = score * 255
    orig_image = orig_image * 255
    orig_image = orig_image.astype(np.uint8)
    vis_img = mark_boundaries(orig_image, mask, color=(1, 0, 0), mode='thick') 

    path = os.path.splitext(path)[0] # Save without extension
    path = path.replace("/","~")
    
    
    plt.axis('off')
    plt.imshow(orig_image, interpolation='none')
    plt.savefig(os.path.join(export_dir, path+'_Orig_Image.png'),bbox_inches='tight', pad_inches=0, dpi=96)
    plt.clf()
    
    plt.axis('off')
    plt.imshow(np_gt, cmap='gray', interpolation='none')
    plt.savefig(os.path.join(export_dir, path+'_GroundTruth.png'),bbox_inches='tight', pad_inches=0, dpi=96)
    plt.clf()
    
    # plt.imshow(heat_map, cmap='jet')
    plt.axis('off')
    plt.imshow(orig_image, cmap='gray', interpolation='none')
    plt.imshow(heat_map, cmap='jet', alpha=0.4, interpolation='none')
    plt.savefig(os.path.join(export_dir, path+'_Predicted heat map.png'),bbox_inches='tight', pad_inches=0, dpi=96)
    plt.clf()

    plt.axis('off')
    plt.imshow(mask, cmap='gray', interpolation='none')
    plt.savefig(os.path.join(export_dir, path+'_Predicted mask optional.png'),bbox_inches='tight', pad_inches=0, dpi=96)
    plt.clf()

    plt.axis('off')
    plt.imshow(vis_img, interpolation='none')
    plt.savefig(os.path.join(export_dir, path+'_Segmentation result.png'),bbox_inches='tight', pad_inches=0, dpi=96)
    plt.close()



def PSAD_export_gradient_maps(model, testloader, optimizer, np_gt, n_batches=1):
    plt.figure(figsize=(10, 10))
    testloader.dataset.get_fixed = True
    cnt = 0
    degrees = -1 * np.arange(c.n_transforms_test) * 360.0 / c.n_transforms_test

    origin_list = list()
    grad_list = list()
    path_list = list()
    # TODO n batches
    for i, data in enumerate(tqdm(testloader, disable=c.hide_tqdm_bar)):
        optimizer.zero_grad()
        inputs, labels = preprocess_batch(data[:2])
        paths = data[2]
        inputs = Variable(inputs, requires_grad=True)

        emb = model(inputs)
        loss = get_loss(emb, model.nf.jacobian(run_forward=False))
        loss.backward()

        grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
        grad = grad[labels > 0]
        if grad.shape[0] == 0:
            continue
        grad = t2np(grad)

        inputs = inputs.view(-1, c.n_transforms_test, *inputs.shape[-3:])[:, 0]
        inputs = inputs[:,:3,:,:] # [1,12,448,448] -> [1,3,448,448]
        inputs = np.transpose(t2np(inputs[labels > 0]), [0, 2, 3, 1]) #[1,448,448,3]
        inputs_unnormed = np.clip(inputs * c.norm_std + c.norm_mean, 0, 1)

        for i_item in range(c.n_transforms_test):
            old_shape = grad[:, i_item].shape
            img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
            img = np.transpose(img, [1, 2, 0])
            img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
            img = gaussian_filter(img, (0, 3, 3))
            grad[:, i_item] = np.reshape(img, old_shape)

        grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
        grad_img = np.mean(np.abs(grad), axis=1) 
        grad_img_sq = grad_img ** 2
        
        
        grad_list.append(grad_img_sq)
        origin_list.append(inputs_unnormed)
        path_list += paths
        
    grad_list = np.concatenate(grad_list,axis=0) 
    origin_list = np.concatenate(origin_list,axis=0)

    
    
    max_score = grad_list.max()
    min_score = grad_list.min()
    grad_list = (grad_list - min_score) / (max_score - min_score)
    
    
    fpr, tpr, _ = roc_curve(np_gt.flatten(), grad_list.flatten())
    per_pixel_rocauc = roc_auc_score(np_gt.flatten(), grad_list.flatten()) 


    # find optimal threshold
    precision, recall, thresholds = precision_recall_curve(np_gt.flatten(), grad_list.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    optimal_threshold = thresholds[np.argmax(f1)]


    print("Localization AUROC : {}".format(per_pixel_rocauc))
    cnt = save_imgs(origin_list, grad_list, np_gt, optimal_threshold, path_list, cnt)


parser = argparse.ArgumentParser(description='choose category')
parser.add_argument('--label', required=True)
args = parser.parse_args()

c.class_name = args.label
c.modelname = 'PSAD_' + args.label



not_implemented_localization = ['multi_early_rgb', 'multi_late_max_pooling', 'multi_late_mean_max_pooling']
if plus_variable.version in not_implemented_localization:
    print(plus_variable.version+'_localization test is not implemented')
    exit()







train_set, test_set = load_datasets(c.dataset_path, c.class_name)
train_set.is_PSAD_eval, test_set.is_PSAD_eval = True, True
train_loader = DataLoader(train_set, pin_memory=True, batch_size=c.batch_size, shuffle=True, num_workers=10,
                                            drop_last=False)
test_loader = DataLoader(test_set, pin_memory=True, batch_size=c.batch_size_test, shuffle=False, num_workers=10,
                                            drop_last=False)



gt_transfomrs = transforms.Compose([
    transforms.Resize(c.img_size, interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

ground_data_path = os.path.join(c.dataset_path,c.class_name,'ground_truth')
ground_dataset = ImageFolder(root=ground_data_path, transform=gt_transfomrs)

if plus_variable.version=='single_rgb': 
    ground_dataset.samples = [[i]*4 for i in ground_dataset.samples] 
    def flatten(lst):
        result = []
        for item in lst:
            result.extend(item)
        return result
    ground_dataset.samples = flatten(ground_dataset.samples)
ground_dataloader = DataLoader(
    ground_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=10,
)

x_ground = next(iter(ground_dataloader))[0].numpy()
ground_temp = x_ground

np_gt = np.transpose(ground_temp, (0, 2, 3, 1))
np_gt = np.mean(np_gt, axis=3)
##



model = load_model(c.modelname)
optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)

model.to(c.device)

## evaluate
model.eval()
PSAD_export_gradient_maps(model, test_loader, optimizer, np_gt, -1)

