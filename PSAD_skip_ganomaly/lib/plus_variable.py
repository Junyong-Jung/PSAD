import PSAD_config

previous_conv_ch = 100 
insert_location = 12



file_dict = {"single_rgb" : ["1.png", "2.png", "3.png", "4.png"],
             "single_normal" : ["normal.png"],
             "single_gx" : ["normal_x.png"],
             "single_gy" : ["normal_y.png"],
             "single_gauss_curvature" : ["Gauss.png"],
             "single_mean_curvature" : ["Mean.png"],
             "multi_early_rgb" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_late_mean_pooling" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_late_max_pooling" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_late_mean_max_pooling" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_channel_attention_rgb" : ["1.png", "2.png", "3.png", "4.png"],
             "multi_channel_attention_rgb+normal" : ["1.png", "2.png", "3.png", "4.png", "normal.png"],
             "multi_channel_attention_albedo+normal" : ["albedo.png", "normal.png"]}



version = PSAD_config.version

file_list = file_dict[version]
num_imgs = len(file_list) if 'single' not in version else 1