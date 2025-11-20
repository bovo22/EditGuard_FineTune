import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    """ initialization for distributed training """
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def cal_pnsr(sr_img, gt_img):
    """calculate PSNR (uint8 numpy 이미지 기준)"""
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
    return psnr


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none',
        help='job launcher'
    )
    parser.add_argument(
        '--ckpt', type=str,
        default='/userhome/NewIBSN/EditGuard_open/checkpoints/clean.pth',
        help='Path to pre-trained model.'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--out_csv', type=str, default=None,
        help='(optional) CSV file path to save per-image metrics'
    )
    args = parser.parse_args()

    # ★ 평가용: is_train=False
    opt = option.parse(args.opt, is_train=False)

    # distributed settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # resume_state는 테스트/평가에서는 보통 사용 안 함
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
        option.check_resume(opt, resume_state['iter'])
    else:
        resume_state = None

    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True

    #### create val dataloader (기존 test.py 로직 재사용)
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        print("phase", phase)
        if phase in ['TD', 'val']:
            val_set = create_dataset(dataset_opt)
            if opt['dist']:
                world_size = torch.distributed.get_world_size()
                train_sampler = DistIterSampler(
                    val_set, world_size, rank, dataset_opt['ratio']
                )
                val_loader = create_dataloader(
                    val_set, dataset_opt, opt, train_sampler
                )
            else:
                val_loader = create_dataloader(
                    val_set, dataset_opt, opt, None
                )
            break

    if val_loader is None:
        raise RuntimeError('No TD/val dataset found in opt[yaml].')

    # create model
    model = create_model(opt)
    model.load_test(args.ckpt)

    # --- metric accumulation ---
    avg_psnr_cover = 0.0        # SR vs GT
    avg_psnr_secret_list = []   # SR_h vs LR_ref
    avg_psnr_stego = 0.0        # LR vs GT
    biterr_list = []
    num_imgs = 0

    # per-image 기록 (CSV 용)
    per_image_metrics = []      # (global_idx, image_id, frame_id, psnr_cover, psnr_stego, bit_err)

    idx = 0
    for image_id, val_data in enumerate(val_loader):
        img_dir = os.path.join('results', opt['name'])
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test(image_id)

        visuals = model.get_current_visuals()

        t_step = visuals['SR'].shape[0]
        idx += t_step
        n = len(visuals['SR_h'])

        a = visuals['recmessage'][0]
        b = visuals['message'][0]

        bitrecord = util.decoded_message_error_rate_batch(a, b)
        biterr_list.append(bitrecord)

        # secret PSNR accumulator 초기화
        if len(avg_psnr_secret_list) == 0:
            avg_psnr_secret_list = [0.0] * n

        for i in range(t_step):
            # uint8 이미지로 변환
            sr_img = util.tensor2img(visuals['SR'][i])
            sr_img_h = []
            for j in range(n):
                sr_img_h.append(util.tensor2img(visuals['SR_h'][j][i]))
            gt_img = util.tensor2img(visuals['GT'][i])
            lr_img = util.tensor2img(visuals['LR'][i])
            lrgt_img = []
            for j in range(n):
                lrgt_img.append(util.tensor2img(visuals['LR_ref'][j][i]))

            # 필요하면 저장 (원 test.py와 동일)
            save_img_path = os.path.join(img_dir, '{:d}_{:d}_{:s}.png'.format(image_id, i, 'SR'))
            util.save_img(sr_img, save_img_path)

            for j in range(n):
                save_img_path = os.path.join(img_dir, '{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'SR_h'))
                util.save_img(sr_img_h[j], save_img_path)

            save_img_path = os.path.join(img_dir, '{:d}_{:d}_{:s}.png'.format(image_id, i, 'GT'))
            util.save_img(gt_img, save_img_path)

            save_img_path = os.path.join(img_dir, '{:d}_{:d}_{:s}.png'.format(image_id, i, 'LR'))
            util.save_img(lr_img, save_img_path)

            for j in range(n):
                save_img_path = os.path.join(img_dir, '{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'LRGT'))
                util.save_img(lrgt_img[j], save_img_path)

            # --- PSNR 계산 ---
            psnr_cover = cal_pnsr(sr_img, gt_img)
            psnr_secret_list = []
            for j in range(n):
                psnr_secret_list.append(cal_pnsr(sr_img_h[j], lrgt_img[j]))
            psnr_stego = cal_pnsr(lr_img, gt_img)

            avg_psnr_cover += psnr_cover
            for j in range(n):
                avg_psnr_secret_list[j] += psnr_secret_list[j]
            avg_psnr_stego += psnr_stego
            num_imgs += 1

            global_idx = len(per_image_metrics)
            per_image_metrics.append((
                global_idx,
                image_id,
                i,
                float(psnr_cover),
                float(psnr_stego),
                float(bitrecord)
            ))

    # --- 평균 계산 ---
    avg_psnr_cover /= num_imgs
    avg_psnr_stego /= num_imgs
    avg_psnr_secret_list = [p / num_imgs for p in avg_psnr_secret_list]
    avg_biterr = sum(biterr_list) / len(biterr_list)

    res_psnr_secret = ''
    for p in avg_psnr_secret_list:
        res_psnr_secret += ('_{:.4e}'.format(p))

    print('# Evaluation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e}, Bit_Error: {:.4e}'.format(
        avg_psnr_cover, res_psnr_secret, avg_psnr_stego, avg_biterr
    ))

    # --- CSV 저장 (옵션) ---
    if args.out_csv is not None:
        print(f'Saving per-image metrics to {args.out_csv}')
        with open(args.out_csv, 'w') as f:
            # per-image 결과
            f.write('idx,image_id,frame_id,psnr_cover,psnr_stego,bit_error\n')
            for row in per_image_metrics:
                idx_, img_id, frame_id, psnr_c, psnr_s, bit_e = row
                f.write(f'{idx_},{img_id},{frame_id},{psnr_c},{psnr_s},{bit_e}\n')

            f.write(
                '# Evaluation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, '
                'PSNR_Stego: {:.4e}, Bit_Error: {:.4e}\n'.format(
                    avg_psnr_cover, res_psnr_secret, avg_psnr_stego, avg_biterr
                )
            )
if __name__ == '__main__':
    main()