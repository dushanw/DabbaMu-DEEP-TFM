def sort_name_by_epoch(x):
    return int(x.split('/')[-1].split('_')[0])

def sort_by_attr_values(img_dir):
    img_dir= img_dir.split('/')[-2]
    values=[]
    for attr in img_dir.split(')'):
        if attr== '':continue
        try:values.append(float(attr.split('(')[1]))
        except:values.append(attr.split('(')[1])
    return values

def heatmap_sort_function(heatmap_dir):
    attrs = heatmap_dir[:-4].split('/')[-1].split('@')[2:]
    attrs.reverse()
    
    metric = heatmap_dir[:-4].split('/')[-1].split('@')[0]
    values = []
    
    for attr in attrs:
        if attr== '':continue
        attr_value= attr.split('(')[1][:-1]
        try:values.append(float(attr_value)) # float values
        except:
            try:values.append(list(map(float, eval(attr_value)))) # list of float
            except:
                try:values.append(list(eval(attr_value))) #list of strings/ etc
                except:values.append(attr_value) # string/ etc values           
                
    values.append(metric)
    values.reverse()
    return values