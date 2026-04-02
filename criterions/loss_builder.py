__all__ = [
    'build_criterion'
]

from utils import Registry

CRITERIONS = Registry()

# def build_criterion(cfg, device=None):
#     criterion = CRITERIONS[cfg['model']['loss']](cfg)
#     return criterion.to(device)

def build_criterion(cfg, device=None):
    lossdict = {}
    losses = cfg['model']['loss'].split('+')
    for l in losses:
        n = 'det' if 'UNIFORM' in l else l
        lossdict[n] = CRITERIONS[l](cfg).to(device)  # 统一 uniform 和 nonuniform
    return lossdict
