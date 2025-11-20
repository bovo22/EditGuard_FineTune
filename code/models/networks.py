import logging
import math

from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet

logger = logging.getLogger('base')

####################
# define network
####################
def define_G_v2(opt):
    # opt 전체 옵션에서 네트워크 관련 부분만 따로 뽑음
    opt_net = opt['network_G']          # options['network_G'] 블록

    # network_G 블록 안의 항목들
    which_model = opt_net['which_model_G']   # 예: 'IBSN' (지금은 안 쓰지만 남겨 둠)
    subnet_type = opt_net['subnet_type']     # 예: 'INV'
    down_num = int(math.log(opt_net['scale'], 2))

    # 이미지 수 (최상단에 num_image: 1 로 설정해 둠)
    num_image = opt.get('num_image', 1)

    if num_image == 1:
        netG = VSN(opt,
                   subnet(subnet_type, 'xavier'),
                   subnet(subnet_type, 'xavier'),
                   down_num)
    else:
        netG = VSN(opt,
                   subnet(subnet_type, 'xavier'),
                   subnet(subnet_type, 'xavier_v2'),
                   down_num)

    return netG
