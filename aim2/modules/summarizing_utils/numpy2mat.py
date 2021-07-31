import glob, os, shutil
import numpy as np
import scipy.io

def zip_heatmap_vals_matlab(exp_set_dir='figs/mnistv12', target_dir= None): # target_dir: some_dir/.../my_zip_file.zip
    if target_dir==None:target_dir= f'{exp_set_dir}/heatmaps/heatmaps_mat_files'
    else:target_dir= target_dir[:-4] # remove ".zip"
    
    try:shutil.rmtree(f'{exp_set_dir}/heatmaps/mat_files')
    except:pass

    os.mkdir(f'{exp_set_dir}/heatmaps/mat_files')

    content = {}
    for npy_dir in glob.glob(f'{exp_set_dir}/heatmaps/*.npy'):
        content['data'] = np.load(npy_dir)
        file_name = npy_dir[:-4].split('/')[-1] + '.mat'
        print(file_name)
        scipy.io.savemat(f'{exp_set_dir}/heatmaps/mat_files/{file_name}', mdict=content)

    mat_dir = f'{exp_set_dir}/heatmaps/mat_files'
    print(f'\n\nZIPPING ::: {mat_dir} -> {target_dir}')

    try:os.remove(f'{target_dir}.zip')
    except:pass
    
    shutil.make_archive(base_name=target_dir ,format='zip', root_dir= mat_dir, base_dir = '.')