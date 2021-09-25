
import matplotlib.pyplot as plt

class squared_decaying_weighted_skips:
    def __init__(self, zero_weight_epoch, epoch_step=1):
        self.zero_weight_epoch= zero_weight_epoch
        self.epoch_step= epoch_step
        
    def __call__(self, lambda_t, epoch):
        epoch= self.epoch_step * (epoch//self.epoch_step)    
            
        if epoch>self.zero_weight_epoch:skip_weight= 0
        else:skip_weight = ((1.0/self.zero_weight_epoch)*epoch-1)**2

        yt_up_support = skip_weight * lambda_t
        return yt_up_support   
    
    def show_plot(self):
        plot_list = []
        for epoch in range(0, int(self.zero_weight_epoch*1.2)):
            plot_list.append(self.__call__(1, epoch))
        plt.plot(plot_list)
        plt.xlabel('epoch')
        plt.ylabel('weight')
        plt.title(f'squared_decaying_weighted_skips:zero_weight_epoch({self.zero_weight_epoch})|epoch_step({self.epoch_step})')
        plt.show()
        



def no_skips(lambda_t, epoch):
    return 0.* lambda_t