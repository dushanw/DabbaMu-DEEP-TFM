
from main_reconstruction import safe_do_exps

name= 'baselineF' #'bbbcHumanMCF7cellsW4_swinIR_v1_counter'

exp_dir= f'../figs/{name}' #'/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/results/aim2/figs'
save_dir_special_root =  f'../figs/{name}'

#!rm -rf $exp_dir
#!mkdir $exp_dir

#!rm -rf $save_dir_special_root
#!mkdir $save_dir_special_root

count_only = False

device = 'cuda:1'

exps = {
        'DATASET.name': ['bbbcHumanMCF7cellsW4'],
    
        'MODEL.MODEL_A.rotation_lambda': ['10000.0'],
        'MODEL.MODEL_A.lambda_scale_factor': ['4'],
        'MODEL.MODEL_H.T': ['4'], 
        'MODEL.MODEL_H.lr_H': ['1.0'],
    
        'DATASET.img_size':  ['256'], #256
        'DATASET.num_samples_train': ['3000'], #'10000'
        'DATASET.batch_size_train': ['32'] #'nn.L1Loss().to(device)', 
}

general_opts= ['TRAIN.show_results_epoch', '5', #5
                'TRAIN.epochs', '150']  #150

safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)
