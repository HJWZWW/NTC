import os
from utils.utils import *
config_path = './config/config_Hyperprior.yaml'
config = read_config(config_path)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
import torch
import torch.optim as optim
from model.Hyperprior import Hyperprior

from datasets.datasets import get_loader, get_test_loader
from datetime import datetime
from eval_Hyperprior import eval_Hyperprior


config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = config['checkpoint']
use_pretrained = config['use_pretrained']
exp_abstract = '{}_{}'.format(config['exp_tag'], config['lambda'])

def train_Hyperprior():
    name = exp_abstract + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    workdir, logger = logger_configuration(exp_abstract, name, phase='train', save_log=True)
    config['logger'] = logger
    model = Hyperprior(config).to(config['device'])
    if use_pretrained:
        # cdf_size = model.update()
        load_weights(model, model_path)
        cur_epoch = config['cur_epoch']
    else:
        cur_epoch = 0

    if not use_pretrained and config.get("context", False):
        for n, p in model.named_parameters():
            p.requires_grad = ("context_model" in n) or ("entropy_fuse" in n)
    else:
        for _, p in model.named_parameters():
            p.requires_grad = True

    train_loader, test_loader = get_loader(config)
    global global_step
    params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.endswith(".quantiles")]
    optimizer_G = optim.Adam(params, lr=config['lr'])

    aux_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.endswith(".quantiles")]
    aux_optimizer = (optim.Adam(aux_params, lr=config['aux_lr'])
                     if aux_params and config['aux_lr'] > 0 else None)

    tot_epoch = config['epoch']
    global_step = 0
    best_loss = float("inf")
    steps_epoch = global_step // train_loader.__len__()
    steps_epoch += cur_epoch
    for epoch in range(steps_epoch, tot_epoch):
        logger.info('======Current epoch %s ======' % epoch)
        logger.info('experiment abstract: ' + exp_abstract + ' gpu id: ' + str(config['gpu_id']))
        logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
        global_step = train_one_epoch_Hyperprior(epoch, model, train_loader, optimizer_G, aux_optimizer, config, logger, global_step)
        loss, psnr, bpp_y, bpp_z = eval_Hyperprior(model, test_loader, logger, config, global_step, cur_epoch, exp_abstract)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if is_best:
            logger.info('Best Model!')
            save_model(model, save_path=workdir + '/checkpoints/epoch{}_best_loss.model'.format(epoch + 1))
        if (epoch + 1) % 1 == 0:
            save_model(model, save_path=workdir + '/checkpoints/epoch{}.model'.format(epoch + 1))


if __name__ == '__main__':
    train_Hyperprior()