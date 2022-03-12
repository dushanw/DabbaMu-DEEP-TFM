
from main_reconstruction_swinIRwCustomUp_wforward import safe_do_exps

name= 'bbbcHumanMCF7cellsW4_swinIRwCustomV2_EvalHtOurUpsample_cropped64'

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

device = 'cuda:1'

exps = {
        'DATASET.name': ['bbbcHumanMCF7cellsW4cropped'],
    
        'MODEL.MODEL_A.rotation_lambda': ['10000.0'],
        'MODEL.MODEL_A.lambda_scale_factor': ['4'],
        'MODEL.MODEL_H.T': ['4'], 
        'MODEL.MODEL_H.lr_H': ['1.0'],
    
        'DATASET.img_size':  ['64'], #256
        'DATASET.num_samples_train': ['3000'], #'10000-> 1400 sec/ epoch' , 3000-> 440 sec/ epoch (18.3hrs/ exp)  
        'DATASET.num_samples_valtest': ['100'],
        'TRAIN.m_inc_proc': ['inc_m_class(epoch_threshold= 80, epoch_steps= 10)']
}

general_opts= ['DATASET.batch_size_train', '10',
               'DATASET.batch_size_valtest', '10',
               'TRAIN.show_results_epoch', '5', #5
                'TRAIN.epochs', '150']  #150

safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)
