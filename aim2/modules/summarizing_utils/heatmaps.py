import itertools
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

from modules.summarizing_utils.sorting_methods import heatmap_sort_function
from modules.summarizing_utils.filter_results_utils import filter_results, get_metric

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter, A3, A2, A1, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def get_override_dict_list(all_overrides_dict):
    override_dict_list = []
    keys_list= []
    list_value_list = []

    for key, value_list in all_overrides_dict.items():
        keys_list.append(key)
        list_value_list.append(value_list)

    override_combinations= [list(s) for s in itertools.product(*list_value_list)]

    for override_list in override_combinations:
        override_dict= {}
        for i in range(len(keys_list)):
            override_dict[keys_list[i]]=override_list[i]
        override_dict_list.append(override_dict)

    return override_dict_list

def create_metric_map(img_list, dict_, metric_name='SSIM', interested_key1= 'T', interested_key2 = 'lambda_scale_factor', override_dict= {}, show_interested_img_names= False, plot_label_dict= None):
    if 'last_converged_' in metric_name: #eg: metric_name: last_converged_MSE -> which means no matter what we have choosen the last converged result. 
        metric_name= metric_name.split('_')[-1]
    
    for key_ in override_dict:
        dict_[key_] = override_dict[key_]
    
    if show_interested_img_names:print('interested attr dict : ', dict_)
        
    interested_imgs = filter_results(img_list, dict_)
    
    if show_interested_img_names:print('interested_imgs : ', '\n'.join(interested_imgs))
    
    ax1_labels = dict_[interested_key1]
    ax2_labels = dict_[interested_key2]
    
    metric_map = np.ones((len(ax1_labels),len(ax2_labels)), dtype=object)
    epoch_map = np.ones((len(ax1_labels),len(ax2_labels)), dtype='int')
    
    def imgdir2metric(img_name):return get_metric(img_name)[metric_name]
    
    for i in range(len(ax1_labels)):
        ax1_vals = ax1_labels[i]
        for j in range(len(ax2_labels)):
            ax2_vals = ax2_labels[j]

            valid_img_dirs=[]
            for img_dir in interested_imgs:
                if f'{interested_key1}({ax1_vals})' in img_dir and f'{interested_key2}({ax2_vals})' in img_dir:valid_img_dirs.append(img_dir)

            assert len(valid_img_dirs)<=1, f"Multiple directories found for same attr combination (multiple values for one entry of heatmap). Check and add entry to 'override_dict'/ No directories found at all !!! -> dirs :: {valid_img_dirs}"
            if len(valid_img_dirs) == 0:
                valid_img_dirs = ['exp is not done']

            #sorted_valid_img_dirs= sorted(valid_img_dirs, key= imgdir2metric)            
            selected_img_dir = valid_img_dirs[0]

            if selected_img_dir != 'exp is not done':
                metric_dict = get_metric(selected_img_dir)
                metric_map[i, j]= metric_dict[metric_name]
                epoch= int(selected_img_dir.split('/')[-1].split('_')[0])
                epoch_map[i, j]= epoch
            else:
                metric_map[i, j] = "NA"       
                epoch_map[i, j]= 0
            
            
            
            
    if plot_label_dict!= None: ### Use given axis labels
        ax1_labels = plot_label_dict[interested_key1]
        ax2_labels = plot_label_dict[interested_key2]
    
    
    metric_map_noNA= metric_map[np.where(metric_map!= 'NA')]
    if len(metric_map_noNA)==0:
        metric_map_noNA= np.array([0.0])
        
    if metric_name in ['SSIM', 'SSIM11', 'SSIM5']:
        metric_map[np.where(metric_map== 'NA')] = (metric_map_noNA).astype('float').min()
    elif metric_name in ['L1Loss', 'MSE', 'last_converged']:
        metric_map[np.where(metric_map== 'NA')] = (metric_map_noNA).astype('float').max()
    else:
        raise NotImplementedError(f"NA handling is not implemented on -> mode : {mode}")
        
    metric_map= metric_map.astype('float')
    return metric_map, epoch_map, ax1_labels, ax2_labels

def combine_heatmap_with_epochs(metric_heatmap, epoch_matrix, epoch2m_func=None):
    assert metric_heatmap.shape== epoch_matrix.shape, 'Metric heatmap and epoch matrix have different shapes !!!'
    
    metric_heatmap= np.round(metric_heatmap, 5) 
    
    h, w= metric_heatmap.shape
    output_map= np.zeros_like(metric_heatmap).astype('str')
    for i in range(h):
        for j in range(w):
            #m = epoch2m_func(epoch_matrix[i, j])
            #entry= f'{metric_heatmap[i, j]} (m:{m})'
            if epoch_matrix[i, j] == 0:
                entry= 'NA'
            else:
                entry= f'{metric_heatmap[i, j]}/{epoch_matrix[i, j]}'
            output_map[i, j]= entry
    return output_map

def epoch2m(epoch):
    epoch_threshold= 1000
    epoch_steps= 10
    m_inc_step= 1

    if epoch<=epoch_threshold:
        m=1
    else:
        m= 1+((epoch-epoch_threshold)//epoch_steps)*m_inc_step
    return m

def plot_heatmap(metric_map_highlrH, metric_map_lowlrH, epochs_map_highlrH, epochs_map_lowlrH, highlrH, lowlrH, vmin, vmax, x_ticks, y_ticks, interested_key1, interested_key2, override_dict, metric_name, save_dir):
    
    print('******************* WARNING !!! :: m_inc_method :: inc_m_class(epoch_threshold= 80, epoch_steps= 10) is assumed !!! ********************')
    if highlrH!= None and lowlrH!=None:
        plt.figure(figsize= (5*len(y_ticks),3))
        plt.subplot(1,2,1)
        #ax = sns.heatmap(metric_map_highlrH, linewidth=0.5, annot=map_highlrH, vmin=vmin, vmax= vmax, fmt= '.5f')
        ax = sns.heatmap(metric_map_highlrH, linewidth=0.5, annot=combine_heatmap_with_epochs(metric_map_highlrH, epochs_map_highlrH, epoch2m), vmin=vmin, vmax= vmax, fmt= "")

        plt.xticks(np.arange(len(y_ticks))+0.5, y_ticks, rotation=0)
        plt.xlabel(interested_key2)
        plt.yticks(np.arange(len(x_ticks))+0.5, x_ticks, rotation=0)
        plt.ylabel(interested_key1)
        plt.title(f'lr(Ht) : {highlrH}')

        plt.subplot(1,2,2)
        #ax = sns.heatmap(metric_map_lowlrH, linewidth=0.5, annot=True, vmin=vmin, vmax= vmax, fmt= '.5f')
        ax = sns.heatmap(metric_map_lowlrH, linewidth=0.5, annot=combine_heatmap_with_epochs(metric_map_lowlrH, epochs_map_lowlrH, epoch2m), vmin=vmin, vmax= vmax, fmt= "")
        plt.xticks(np.arange(len(y_ticks))+0.5, y_ticks, rotation=0)
        plt.xlabel(interested_key2)
        plt.yticks(np.arange(len(x_ticks))+0.5, x_ticks, rotation=0)
        plt.ylabel(interested_key1)
        plt.title(f'lr(Ht) : {lowlrH}')

        suptit = f'{metric_name} -- '
        overrides= ''
        for key_, val_ in override_dict.items():
            suptit+=f'{key_} : {val_} | '
            overrides += f'{key_}({val_})@'

        suptit = suptit[:-3] 
        overrides= overrides[:-1]

        plt.suptitle(suptit, y= 1.1)

        if save_dir!=None:
            save_dir = f'{save_dir}/heatmaps'

            try:os.mkdir(save_dir)
            except:pass

            np.save(f'{save_dir}/{metric_name}@@{overrides}@@highlrH.npy', [metric_map_highlrH, epochs_map_highlrH])
            np.save(f'{save_dir}/{metric_name}@@{overrides}@@lowlrH.npy', [metric_map_lowlrH, epochs_map_lowlrH])

            plt.savefig(f'{save_dir}/{metric_name}@@{overrides}.jpg', bbox_inches='tight')
        plt.show()
    else:
        
        assert (metric_map_highlrH== metric_map_lowlrH).all(), 'lr_H not specified and different lr_Hs are found !!!'
        
        plt.figure(figsize= (2*len(y_ticks),3))
        #ax = sns.heatmap(metric_map_highlrH, linewidth=0.5, annot=True, vmin=vmin, vmax= vmax, fmt= '.5f')
        ax = sns.heatmap(metric_map_highlrH, linewidth=0.5, annot=combine_heatmap_with_epochs(metric_map_highlrH, epochs_map_highlrH, epoch2m), vmin=vmin, vmax= vmax, fmt= "")

        plt.xticks(np.arange(len(y_ticks))+0.5, y_ticks, rotation=0)
        plt.xlabel(interested_key2)
        plt.yticks(np.arange(len(x_ticks))+0.5, x_ticks, rotation=0)
        plt.ylabel(interested_key1)

        suptit = f'{metric_name} -- '
        overrides= ''
        for key_, val_ in override_dict.items():
            suptit+=f'{key_} : {val_} | '
            overrides += f'{key_}({val_})@'

        suptit = suptit[:-3] 
        overrides= overrides[:-1]

        plt.title(suptit, y= 1.1)

        if save_dir!=None:
            save_dir = f'{save_dir}/heatmaps'

            try:os.mkdir(save_dir)
            except:pass

            np.save(f'{save_dir}/{metric_name}@@{overrides}.npy', [metric_map_highlrH, epochs_map_highlrH])

            plt.savefig(f'{save_dir}/{metric_name}@@{overrides}.jpg', bbox_inches='tight')
        plt.show()
    
    
def plot_all_heat_maps(img_list, attr_dict_highlrH, attr_dict_lowlrH, interested_key1, interested_key2, override_dict, metric_name= 'SSIM', save_dir =None, show_interested_img_names= False, plot_label_dict= None):

    map_metric_highlrH, map_epochs_highlrH, xticks_highlrH, yticks_highlrH = create_metric_map(img_list, attr_dict_highlrH, metric_name=metric_name, interested_key1= interested_key1, interested_key2 = interested_key2, override_dict= override_dict, show_interested_img_names= show_interested_img_names, plot_label_dict= plot_label_dict)
    map_metric_lowlrH, map_epochs_lowlrH, xticks_lowlrH, yticks_lowlrH = create_metric_map(img_list, attr_dict_lowlrH, metric_name=metric_name, interested_key1= interested_key1, interested_key2 = interested_key2, override_dict= override_dict, show_interested_img_names= show_interested_img_names, plot_label_dict= plot_label_dict)

    
    assert xticks_highlrH== xticks_lowlrH, 'Missing lowLR/ highHR images'
    assert yticks_highlrH== yticks_lowlrH, 'Missing lowLR/ highHR images'

    vmin = min(map_metric_highlrH.min(), map_metric_lowlrH.min())
    vmax =  max(map_metric_highlrH.max(), map_metric_lowlrH.max())

    if 'lr_H' in attr_dict_highlrH and 'lr_H' in attr_dict_lowlrH :
        highlrH = attr_dict_highlrH['lr_H']
        lowlrH= attr_dict_lowlrH['lr_H']
    else:
        highlrH, lowlrH= None, None
    
    plot_heatmap(map_metric_highlrH, map_metric_lowlrH, map_epochs_highlrH, map_epochs_lowlrH, highlrH, lowlrH, vmin, vmax, xticks_highlrH, yticks_highlrH, interested_key1, interested_key2, override_dict, metric_name, save_dir)
    
    
    
def quantitative_results_HEATMAPS(img_list, plot_vars_dict, overrides_dict_list, mode, save_dir, show_interested_img_names= False, low_lr_H= '0.0', high_lr_H= '1.0', plot_label_dict= None):
    plot_axes_variables= list(plot_vars_dict.keys())
    
    attr_dict_lowlrH = plot_vars_dict.copy()
    attr_dict_highlrH = plot_vars_dict.copy()
    
    if high_lr_H!= None and  low_lr_H!=None:
        attr_dict_lowlrH['lr_H']= [low_lr_H]
        attr_dict_highlrH['lr_H']= [high_lr_H]
    
        print(f'attr_dict_highlrH : {attr_dict_highlrH}')
        print(f'attr_dict_lowlrH : {attr_dict_lowlrH}')
        
    else:
        print(f'highlrH : {high_lr_H} | lowlrH : {low_lr_H}')
        
    print('ploting heatmaps ... ')
    
    for override_dict in overrides_dict_list:
        print(f'override dict : {override_dict}')

        plot_all_heat_maps(img_list, attr_dict_highlrH, attr_dict_lowlrH, interested_key1 = plot_axes_variables[0], interested_key2= plot_axes_variables[1], override_dict=override_dict, metric_name= mode, save_dir =save_dir, show_interested_img_names= show_interested_img_names, plot_label_dict= plot_label_dict)
    print('ploting heatmaps finished !!!')
    
    if save_dir !=None:
        exp_set_name= save_dir.split('/')[-1]
        heatmap_pdf_filename = f'{save_dir}/heatmaps/heatmaps_{exp_set_name}.pdf'
        save_heatmaps2pdf(f'{save_dir}/heatmaps', heatmap_sort_function, heatmap_pdf_filename)
        print(f'quantitative results saved : {heatmap_pdf_filename}')
    else:
        print('quantitative results not saved !!!')


def save_heatmaps2pdf(heatmap_dir, heatmap_sort_function, pdf_filename):
    heatmap_dir_list= glob.glob(f'{heatmap_dir}/*.jpg')
    heatmap_dir_list_sorted = sorted(heatmap_dir_list, key = heatmap_sort_function)
    doc = SimpleDocTemplate(pdf_filename,pagesize=landscape(A1),
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)
    Story=[]
    
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        

    for idx in range(len(heatmap_dir_list_sorted)):
        if idx!=0 and idx%3==0:Story.append(PageBreak())
        img_name= heatmap_dir_list_sorted[idx]
        
        ptext = '<font size="12">%s</font>' % img_name.split('/')[-1]
        Story.append(Paragraph(ptext, styles["Normal"])) 
        Story.append(Spacer(1, 10))
        
        im = Image(img_name) #8*inch, 2*inch
        Story.append(im)
        Story.append(Spacer(1, 30))
        
    doc.build(Story)
    
    