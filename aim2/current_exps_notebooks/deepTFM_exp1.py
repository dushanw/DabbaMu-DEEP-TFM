

name= 'bloodvesselsDeepTFM6sls_v4' #'bbbcHumanMCF7cellsW4_swinIR_v1_counter'

exp_dir= f'../figs/{name}' #'/n/holylfs/LABS/wadduwage_lab/Lab/uom_Udith/results/aim2/figs'
save_dir_special_root =  f'../figs/{name}'


count_only = False

device = 'cuda:0'

exps = {
        'DATASET.name': ['bloodvesselsDeepTFM6sls'],
    
        'MODEL.MODEL_A.noise': ['False'],
    
        'DATASET.img_size':  ['256'], #256
        'DATASET.num_samples_train': ['996'], #'996'
        'DATASET.num_samples_valtest': ['96'], #'96'
        'DATASET.batch_size_train': ['4'],
        'DATASET.batch_size_valtest': ['4'],
        
}


general_opts= ['MODEL.MODEL_A.sPSF', 'load_psf_from_npy("../psfs/psfs_bloodvesselsDeepTFM6sls_spsf.npy")',
               'MODEL.MODEL_A.exPSF', 'load_psf_from_npy("../psfs/psfs_bloodvesselsDeepTFM6sls_expsf.npy")',
               'MODEL.MODEL_A.emPSF', 'load_psf_from_npy("../psfs/psfs_bloodvesselsDeepTFM6sls_empsf.npy")',
               'TRAIN.show_results_epoch', '5', #5
               'TRAIN.epochs', '150',
               'GENERAL.other_opt_dir', 'replace !!!']  #150



from main_reconstruction_swinIR import safe_do_exps
exps['MODEL.MODEL_A.lambda_scale_factor']= ['3'] # 4, 3
exps['MODEL.MODEL_H.T']= ['32']  # 32, 16
exps['MODEL.MODEL_H.lr_H']= ['0.0'] #1.0, 0.0
#general_opts[11]= 'adversarial_learning/swinIRwforward_support_files/opt_16down.yaml'
general_opts[11]= 'adversarial_learning/swinIR_support_files/opt.yaml'
safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)


from main_reconstruction_swinIR_wforward import safe_do_exps
exps['MODEL.MODEL_A.lambda_scale_factor']= ['3'] # 4, 3
exps['MODEL.MODEL_H.T']= ['64']  # 32, 16
exps['MODEL.MODEL_H.lr_H']= ['1.0'] #1.0, 0.0
#general_opts[11]= 'adversarial_learning/swinIRwforward_support_files/opt_16down.yaml'
general_opts[11]= 'adversarial_learning/swinIR_support_files/opt.yaml'
safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)


#from main_reconstruction_swinIR_wforward import safe_do_exps
#exps['MODEL.MODEL_A.lambda_scale_factor']= ['4'] # 4, 3
#exps['MODEL.MODEL_H.T']= ['32']  # 32, 16
#exps['MODEL.MODEL_H.lr_H']= ['1.0'] #1.0, 0.0
#general_opts[11]= 'adversarial_learning/swinIRwforward_support_files/opt_64down.yaml'
#safe_do_exps(exps, general_opts, device, exp_dir = exp_dir, save_special= True, count_only= count_only, save_dir_special_root= save_dir_special_root)

