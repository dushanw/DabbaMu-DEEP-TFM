

def inc_m(m, epoch, n=None):
    if n==None:n=1
    m+=n
    return m

def inc_1_after_60_interval_10(m, epoch):
    if epoch>60 and epoch%10==0:
        m=inc_m(m, epoch, 1)
    return m

class inc_m_class:
    def __init__(self, epoch_threshold, epoch_steps):
        self.epoch_threshold= epoch_threshold
        self.epoch_steps= epoch_steps
    
    def __call__(self, m, epoch):
        if epoch> self.epoch_threshold and epoch% self.epoch_steps==0:
            m=inc_m(m, epoch, 1)
        return m
                

def inc_1_after_10_interval_1(m, epoch):
    if epoch>10 and epoch%1==0:
        m=inc_m(m, epoch, 1)
    return m