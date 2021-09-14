

def decaying_weighted_skips(lambda_t, epoch):
    if epoch>100:skip_weight= 0
    else:skip_weight = (0.01*epoch-1)**2
        
    yt_up_support = skip_weight * lambda_t
    return yt_up_support

def no_skips(lambda_t, epoch):
    return 0.* lambda_t