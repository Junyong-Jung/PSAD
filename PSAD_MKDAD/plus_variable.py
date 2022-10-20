from utils.utils import get_config


config = get_config("configs/config.yaml")


    
file_dict = {"single_rgb" : ["1.png", "2.png", "3.png", "4.png"],
             "single_normal" : ["normal.png"],
             "single_gx" : ["normal_x.png"],
             "single_gy" : ["normal_y.png"],
             "single_gauss_curvature" : ["Gauss.png"],
             "single_mean_curvature" : ["Mean.png"],
             "multi_early_rgb" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_early_rgb+normal" : ["1.png", "2.png", "3.png", "4.png", "normal.png"],
             "multi_early_albedo+normal" : ["albedo.png", "normal.png"],
             "multi_late_mean_pooling" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_late_max_pooling" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_late_mean_max_pooling" : ["1.png", "2.png", "3.png", "4.png"]}

# num_imgs_dict = {"multi_rgb" : 4,
#                 "multi_rgb+normal" : 5,
#                 "multi_albedo+normal" : 2,
#                 "multi_late_mean_pooling" : 4,
#                 "multi_late_max_pooling" : 4,
#                 "multi_late_mean_max_pooling" : 4}


version = config["version"]

file_list = file_dict[version]
num_imgs = len(file_list) if 'single' not in version else 1
# num_imgs = num_imgs_dict[version] if version in num_imgs_dict else None