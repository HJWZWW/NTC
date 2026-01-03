import yaml
import torch
import numpy as np
import time
import logging
import os
import random

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_weights(net, model_path, cdf_size=None):
    pretrained = torch.load(model_path)  # ['state_dict']
    result_dict = {}
    for key, weight in pretrained.items():
        result_dict[key] = weight
    result_dict = {k: v for k, v in result_dict.items() if 'rate_adaption.mask' not in k}
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_train_randomness():
    # when training, seed isn't be fixed
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed()
    np.random.seed()
    torch.manual_seed(random.randint(0, 2**32))


def set_test_randomness(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def logger_configuration(exp_abstract, filename, phase, save_log=True):
    logger = logging.getLogger(exp_abstract)  # logger name
    workdir = './history/{}'.format(filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/Log_{}.log'.format(filename)
    checkpoints = workdir + '/checkpoints'
    makedirs(workdir)
    makedirs(checkpoints)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def calPSNR(img1, img2, max_val=255.):
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def train_one_epoch_Hyperprior(epoch, model, train_loader, optimizer_G, aux_optimizer, config, logger, global_step):
    model.train()
    elapsed, losses, psnrs, bpps_y, bpps_z = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, psnrs, bpps_y, bpps_z]
    for batch_idx, input_image in enumerate(train_loader):
        global_step += 1
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()
        start_time = time.time()
        input_image = input_image.to(config['device'])
        mse_loss, bpp_y, bpp_z, x_hat_ntc = model(input_image)
        loss = mse_loss + config['lambda'] * (bpp_y + bpp_z)

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)

        optimizer_G.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        bpps_y.update(bpp_y.item())
        bpps_z.update(bpp_z.item())

        psnr = 10 * (torch.log10((255 ** 2) / mse_loss))
        psnrs.update(psnr.item())

        if (global_step % config['print_step']) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR_Hyperprior {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'bpp_y {bpps_y.val:.4f}({bpps_y.avg:.4f})',
                f'bpp_z {bpps_z.val:.4f}({bpps_z.avg:.4f})',
                f'Epoch {epoch}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()
    return global_step