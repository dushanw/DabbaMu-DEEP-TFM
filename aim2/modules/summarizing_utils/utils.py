import numpy as np
import matplotlib.pyplot as plt


def show_results(key, dict_img_position, interested_imgs, sort_by_attr_values=None):
    start, end = dict_img_position[key]

    for img_dir in sorted(interested_imgs, key=sort_by_attr_values):
        plt.figure(figsize = (15,5))
        plt.imshow(plt.imread(img_dir)[start:end,:])
        plt.title(img_dir)
        plt.show()
        

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points