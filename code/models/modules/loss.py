import torch
import torch.nn as nn
import numpy as np


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super().__init__()

        # 1) None 방어
        if losstype is None:
            losstype = 'l2'

        # 2) 소문자로 정규화
        if isinstance(losstype, str):
            losstype = losstype.lower()

        # 3) alias 정리
        alias = {
            'mse': 'l2',
            'mse_loss': 'l2',
            'mae': 'l1',
        }
        self.losstype = alias.get(losstype, losstype)
        self.eps = eps
    def forward(self, x, target):
        # l2 / mse 계열
        if self.losstype in ['l2', 'mse', 'mse_loss']:
            loss = torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))

        # l1 / mae 계열
        elif self.losstype in ['l1', 'mae']:
            diff = x - target
            loss = torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))

        # center 옵션
        elif self.losstype == 'center':
            # batch-wise 벡터를 유지하고 싶으면 mean 없이 사용
            loss = torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))

        else:
            print(f"reconstruction loss type error! (losstype: {self.losstype})")
            # 이상한 값 들어오면 기본적으로 l2 로 처리
            loss = torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))

        # ★ 항상 torch.Tensor 반환
        return loss

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss
    

class ReconstructionMsgLoss(nn.Module):
    def __init__(self, losstype='mse'):
        super(ReconstructionMsgLoss, self).__init__()
        self.losstype = losstype
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, messages, decoded_messages): 
        if self.losstype == 'mse':
            return self.mse_loss(messages, decoded_messages)
        elif self.losstype == 'bce':
            return self.bce_loss(messages, decoded_messages)
        elif self.losstype == 'bce_logits':
            return self.bce_logits_loss(messages, decoded_messages)
        else:
            print("ReconstructionMsgLoss loss type error!")
            return 0
