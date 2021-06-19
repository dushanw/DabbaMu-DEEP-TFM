import numpy as np
import matplotlib.pyplot as plt
import glob
import os


import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

def get_available_attr(img_list):
    attr_dict = {}
    for img_dir in img_list:
        for attr in img_dir.split('/')[-2].split('@'):
            attr_name, attr_value = attr.split('(')[0], attr.split('(')[1][:-1]
            if attr_name not in attr_dict.keys():
                attr_dict[attr_name] = [attr_value]
            else:
                if attr_value not in attr_dict[attr_name]: 
                    attr_dict[attr_name].append(attr_value)
    return attr_dict


def filter_results(attr_dict, img_list=None):
    interested_imgs = []
    attrs = []
    attr_names = list(attr_dict.keys())
    attr_list= attr_dict.values()
    
    param_combination_list = cartesian_coord(*attr_list)
    
    for i in range(len(param_combination_list)):
        param_comb= param_combination_list[i]
        
        attrs= []
        for j in range(len(param_comb)):
            attr_name = attr_names[j]
            param_value = param_comb[j]
            attrs.append(f'{attr_name}({param_value})')
            
        #print(attrs)
        for img_dir in img_list:
            flag=True
            for attr in attrs:
                #print('check : ',attr.split('('))
                if attr.split('(')[1][:-1]=='all':
                    flag=True
                    continue
                else:
                    if attr not in img_dir:
                        flag= False
                        break
            if flag==True:interested_imgs.append(img_dir)
            #else:print(img_dir)
    print(f'{len(interested_imgs)} images are found !!!')
    return interested_imgs

def show_results(key, dict_img_position, interested_imgs, sort_by_attr_values=None):
    start, end = dict_img_position[key]

    for img_dir in sorted(interested_imgs, key=sort_by_attr_values):
        plt.figure(figsize = (15,5))
        plt.imshow(plt.imread(img_dir)[start:end,:])
        plt.title(img_dir)
        plt.show()

def sort_name_by_epoch(x):
    return int(x.split('/')[-1].split('_')[0])

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

def find_best_result(img_dir):
    img_list = sorted(glob.glob(f"{img_dir}/*.jpg"),key= sort_name_by_epoch, reverse=True)
    min_loss=1000
    final_img_dir= None
    
    losses=[]
    for img_dir in img_list:
        loss= img_dir.split('(')[-1][:-5]
        if loss!='nan':losses.append(float(loss))
        
    min_loss= min(losses)
    
    for img_dir in img_list:
        loss= img_dir.split('(')[-1][:-5]
        
        img= plt.imread(img_dir)
        is_results_okay= img[100, 300].sum()< 765
        

        if is_results_okay and loss!='nan':
            loss= float(loss)
            if loss<min_loss+0.005:
                return img_dir
                
            
    print(f'####   no image found : {img_dir}')
    return None

def get_img_list(img_dir = 'figs/mnistv6', mode='lowest_loss', loss_threshold=0.05):
    exp_list = sorted(glob.glob(f'{img_dir}/*@*'))
    
    img_dirs=[]
    for idx in range(len(exp_list)):
        #if idx>102:break
        exp_dir = exp_list[idx]
            
        if mode=='last_converged':img_dir = find_last_converged_result(exp_dir, loss_threshold)
        elif mode=='lowest_loss':img_dir = find_best_result(exp_dir)
        
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
    
    
def sort_by_attr_values(img_dir):
    img_dir= img_dir.split('/')[-2]
    values=[]
    for attr in img_dir.split(')'):
        if attr== '':continue
        values.append(float(attr.split('(')[1]))
    return values

