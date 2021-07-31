
import glob
import matplotlib.pyplot as plt

from modules.summarizing_utils.sorting_methods import sort_name_by_epoch
from modules.summarizing_utils.filter_results_utils import get_metric


def find_last_converged_result(img_dir, loss_threshold=0.05):
    img_list = sorted(glob.glob(f"{img_dir}/*.jpg"),key= sort_name_by_epoch, reverse=True)
    for img_dir in img_list:
        loss= float(img_dir.split('(')[-1][:-5])
        
        img= plt.imread(img_dir)
        is_loss_okay= loss< loss_threshold
        is_results_okay= img[100, 300].sum()< 765

        if is_loss_okay and is_results_okay:
            return img_dir
    
    print(f'####   no image found : {img_dir}')
    return None


def find_best_result(img_dir, metric_name='SSIM', metric_type= 'score'): #metric_type= loss/ score
    img_list = sorted(glob.glob(f"{img_dir}/*.jpg"),key= sort_name_by_epoch, reverse=True)
    min_loss=1000
    final_img_dir= None
    
    metric_list=[]
    for img_dir in img_list:
        metric_dict = get_metric(img_dir)
        metric = metric_dict[metric_name]
        if metric!='nan':metric_list.append(metric)
        
    min_metric= min(metric_list)
    max_metric= max(metric_list)
    
    for img_dir in img_list:
        metric_dict = get_metric(img_dir)
        metric = metric_dict[metric_name]
        
        img= plt.imread(img_dir)
        is_results_okay= img[100, 300].sum()< 765
        

        if is_results_okay and metric!='nan':
            #loss= float(loss)
            if metric_type== 'loss':
                if metric<min_metric+0.005:
                    return img_dir
            elif metric_type== 'score':
                if metric>max_metric-0.005:
                    return img_dir
                
            
    print(f'####   no image found : {img_dir}')
    return None


def get_img_list(img_dir = 'figs/mnistv6', mode='L1Loss', loss_threshold=0.05):
    exp_list = sorted(glob.glob(f'{img_dir}/*@*'))
    
    img_dirs=[]
    for idx in range(len(exp_list)):
        #if idx>102:break
        exp_dir = exp_list[idx]
            
        if mode=='last_converged':img_dir = find_last_converged_result(exp_dir, loss_threshold)
        elif mode=='L1Loss':img_dir = find_best_result(exp_dir, metric_name='L1Loss', metric_type= 'loss')
        elif mode=='MSE':img_dir = find_best_result(exp_dir, metric_name='MSE', metric_type= 'loss')
        elif mode=='SSIM':img_dir = find_best_result(exp_dir, metric_name='SSIM', metric_type= 'score')
        elif mode=='SSIM5':img_dir = find_best_result(exp_dir, metric_name='SSIM5', metric_type= 'score')
        elif mode=='SSIM11':img_dir = find_best_result(exp_dir, metric_name='SSIM11', metric_type= 'score')
        
        if idx%100==0:
            print(f'{idx+1}/{len(exp_list)} : {img_dir}')
        
        # exceptions
        #if idx==343:img_dir = find_last_converged_result(exp_dir, 0.130)
        ##
        
        if img_dir==None:
            continue
            
        
        img_dirs.append(img_dir)
    print(f'len img dirs : {len(img_dirs)}')
    return img_dirs
    
    