#random_illum [lr_H= 0.0]
#from main_reconstruction_swinIR import safe_do_exps # their_upsample [T=1, lambda_s= 3]
#lr_H, T, lambda_s, batch_size= ['0.0', '1', '3', '128']
#from main_reconstruction_swinIRwCustomUp import safe_do_exps # our_upsample [T=4, lambda_s= 4]
#lr_H, T, lambda_s, batch_size= ['0.0', '4', '4', '2']


#learned illum [lr_H= 1.0] -> might be too high [DO EXPS TO FIND PROPER LR RANGE !!!]
#from main_reconstruction_swinIR_wforward import safe_do_exps # their_upsample [T=1, lambda_s= 3]
#lr_H, T, lambda_s, batch_size= ['1.0', '1', '3', '128']
from main_reconstruction_swinIRwCustomUp_wforward import safe_do_exps # our_upsample [T=4, lambda_s= 4]
lr_H, T, lambda_s, batch_size= ['1.0', '4', '4', '10']


name= 'div2kflickr2k_learnableOurUp'
device = 'cuda:0'

###################################

exps = {
        'DATASET.name': ['div2kflickr2k'], #fixed
    
        'MODEL.MODEL_A.rotation_lambda': ['10000.0'], #fixed
        'MODEL.MODEL_H.lr_H': [lr_H],
        'MODEL.MODEL_H.T': [T], 
        'MODEL.MODEL_A.lambda_scale_factor': [lambda_s],
    
        'DATASET.img_size':  ['64'],  #fixed
        'DATASET.num_samples_train': ['3450'],  #'3450' fixed
        'DATASET.num_samples_valtest': ['100'], #'100' fixed
}

general_opts=  ['DATASET.batch_size_valtest', batch_size,
                'DATASET.batch_size_train', batch_size,
                'TRAIN.show_results_epoch', '5', #5  #fixed
                'TRAIN.epochs', '150']  #150  #fixed


exp_dir= f'../figs/{name}' #'/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/results/aim2/figs'
save_dir_special_root = f'../figs/{name}' #f'/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/results/aim2/figs/{name}'

import shutil, os
try:shutil.rmtree(exp_dir)
except:pass
os.mkdir(exp_dir)
try:shutil.rmtree(save_dir_special_root)
except:pass
os.mkdir(save_dir_special_root)

count_only = False
safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)
