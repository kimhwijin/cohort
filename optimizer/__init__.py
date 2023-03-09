from .builder import build_adam, build_sgd


def get_optimizer(name, config, parameters):
    if name == 'adam':
        return build_adam(config, parameters)
    elif name == 'sgd':
        return build_sgd(config, parameters)