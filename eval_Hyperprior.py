from utils.utils import *
from datasets.datasets import get_loader, get_test_loader
from datetime import datetime
from loss.distortion import Distortion
from model.Hyperprior import Hyperprior
import random, numpy as np, torch
seed = 0
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def eval_Hyperprior(model, test_loader, logger, config, global_step, cur_epoch, exp_abstract):
    # set_test_randomness()
    with torch.no_grad():
        model.eval()
        elapsed, losses, psnrs, bpps_y, bpps_z = [AverageMeter() for _ in range(5)]
        PSNR_list = []
        for batch_idx, input_image in enumerate(test_loader):
            # if batch_idx == 18:
            #     print('debug')
            start_time = time.time()
            input_image = input_image.cuda()
            mse_loss, bpp_y, bpp_z, x_hat_Hyperprior = model(input_image)

            loss = mse_loss + config['lambda'] * (bpp_y + bpp_z)
            losses.update(loss.item())
            bpps_y.update(bpp_y)
            bpps_z.update(bpp_z)
            elapsed.update(time.time() - start_time)

            psnr = calPSNR(input_image, x_hat_Hyperprior).mean()

            psnrs.update(psnr)

            PSNR_list.append(psnr)


        import torchvision.transforms as transforms

        # Extract the first image from x_hat_Hyperprior
        first_image_tensor = x_hat_Hyperprior[0]

        # Convert the tensor to a PIL image
        to_pil = transforms.ToPILImage()
        first_image_pil = to_pil(first_image_tensor.cpu())


        result_root = './results/results_' + exp_abstract

        makedirs(result_root)
        first_image_pil.save(
            result_root + '/epoch{}.png'.format(
                str(cur_epoch + int(global_step / (100000 / config['batch_size'])))))
        logger.info(f'Finish test! Loss={losses.avg:.2f}, Average PSNR={psnrs.avg:.4f}dB, Average bpp_y={bpps_y.avg:.5f}, Average bpp_z={bpps_z.avg:.5f}')
        return losses.avg, psnrs.avg, bpps_y.avg, bpps_z.avg


if __name__ == '__main__':

    config_path = './config/config_Hyperprior.yaml'
    config = read_config(config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = config['checkpoint']
    use_pretrained = config['use_pretrained']

    exp_abstract = '{}_{}'.format(config['exp_tag'], config['lambda'])

    name = exp_abstract + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S_')
    workdir, logger = logger_configuration(exp_abstract, name, phase='test', save_log=True)
    config['logger'] = logger
    model = Hyperprior(config).to(config['device'])
    model.eval()
    # model.update()
    load_weights(model, model_path)
    cur_epoch = config['cur_epoch']

    train_loader, test_loader = get_loader(config)
    global_step = 0
    eval_Hyperprior(model, test_loader, logger, config, global_step, cur_epoch,
                                                          exp_abstract)