 
def do_exps(exps_dict= None, general_opts= None, device= None, exp_dir = '../figs/test', save_special= False):    
    exp_idx = 0
    keys= list(exps.keys())
    
    val_list_list= []
    key_list = [] #eg: 'MODEL.MODEL_A.rotation_lambda'
    key_suffix_list= [] # eg: 'rotation_lambda'
    
    for key, val_list in exps_dict.items():
        key_list.append(key)
        key_suffix_list.append(key.split('.')[-1])
        
        val_list_list.append(val_list)
        
    attr_combination_list = [list(s) for s in itertools.product(*val_list_list)]
    
    print(f'number of total experiments : {len(attr_combination_list)}')
    
    count_already_trained=0
    count_train_from_begining=0
    for attr_combination in attr_combination_list:
        save_dir = f'{exp_dir}/'
        opts= []
        for idx in range(len(attr_combination)):
            opts += [key_list[idx], attr_combination[idx]]
            attr= attr_combination[idx]
            
            #####
            attr_is_list= False
            try:attr_is_list = isinstance(eval(attr), list)
            except:pass
            if attr_is_list:
                attr= '_'.join(list(map(str, eval(attr)))) ## [, ] should not be in the directory name because glob is sensitive to that !
            #####
            
            save_dir+= f'{key_suffix_list[idx]}({attr})@'
    
        save_dir = save_dir[:-1] # remove last '@' 

        exp_idx+=1
        opts_other= ['NAME', f'exp_idx({exp_idx})', 
                     'GENERAL.device', device, 
                     'GENERAL.save_dir', save_dir
                    ]
                     
        opts_other+= general_opts

        opts = opts_other + opts

        
        if len(glob.glob(f'{save_dir}/1_*.jpg'))!=0:
            count_already_trained+=1
            print(f'PASSING :: {save_dir}')
            continue
        
        count_train_from_begining+=1
                    
        try:shutil.rmtree(save_dir)
        except:pass

        save_folder_name= save_dir.split('/')[-1]
        if len(save_folder_name)>255:
            print(f'\nFolder length is too long: len(results_saving_folder) -> {len(save_folder_name)} (<= 255)')
            print(save_folder_name)
        
        run(opts= opts, save_special=save_special)
        #try:
        #    run(opts= opts, save_special=save_special)
        #except Exception as e:
        #    error_file_name = f'{exp_dir}/errors.txt'
        #    write_errormsg2file(f'ERROR : {save_dir}\n {e} \n\n', error_file_name)
        #    print(f'ERROR : {save_dir}\n {e} \n\n')
    print('count_already_trained : ', count_already_trained)
    print('count_train_from_begining : ', count_train_from_begining)
    