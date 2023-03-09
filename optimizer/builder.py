from torch import optim

def build_adam(config, parameters):
    if hasattr(config, 'betas'):
        betas = config.betas
    else:
        betas = (0.9, 0.999)
    
    if hasattr(config, 'weight_decay'):
        weight_decay = config.weight_decay
    else:
        weight_decay = 0.

    return optim.Adam(params=parameters, 
               lr=config.lr, 
               betas=betas, 
               weight_decay=weight_decay)

def build_sgd(config, parameters):
    if hasattr(config, 'monentum'):
        momentum = config.momentum
    else:
        momentum = 0.9

    if hasattr(config, 'weight_decay'):
        weight_decay = config.weight_decay
    else:
        weight_decay = 0.

    if hasattr(config, 'nesterov'):
        nesterov = config.nesterov
    else:
        nesterov = False

    return optim.SGD(params=parameters,
              lr=config.lr,
              momentum=momentum,
              weight_decay=weight_decay,
              nesterov=nesterov
              )