import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR

def build_optimizer(model, config):
    name = config['trainer']['optimizer']
    lr = config['trainer']['true_lr']

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['trainer']['adam_decay'])
    elif name == "adamw":
        # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['trainer']['adamw_decay'])
    # elif name == "sgd":
    #     return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # elif name == "ranger_lars":
    #     return RangerLars(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-6, weight_decay=wd)
    # elif name == "lamb":
    #     return Lamb(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config['trainer']['scheduler_invervel']}
    name = config['trainer']['scheduler']

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config['trainer']['mslr_milestones'], gamma=config['trainer']['mslr_gamma'])})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config['trainer']['cosa_tmax'])})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config['trainer']['elr_gamma'])})
    else:
        raise NotImplementedError()

    return scheduler
