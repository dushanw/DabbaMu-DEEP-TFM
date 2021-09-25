from modules.summarizing_utils.utils import cartesian_coord

def get_metric(img_dir):
    img_name = img_dir.split('/')[-1]
    metrics = img_name[:-4].split('_')[1:]
    metric_dict = {}
    for metric in metrics:
        metric_value = metric.split('(')[1][:-1]
        if metric_value!='nan':metric_value=float(metric_value)
            
        metric_dict[metric.split('(')[0]] = metric_value
    return metric_dict

def get_available_attr(img_list):
    attr_dict = {}
    for img_dir in img_list:
        for attr in img_dir.split('/')[-2].split('@'):
            attr_name, attr_value = attr.split('(')[0], '('.join(attr.split('(')[1:])[:-1]
            if attr_name not in attr_dict.keys():
                attr_dict[attr_name] = [attr_value]
            else:
                if attr_value not in attr_dict[attr_name]: 
                    attr_dict[attr_name].append(attr_value)
    return attr_dict


def filter_results(img_list, attr_dict):
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

