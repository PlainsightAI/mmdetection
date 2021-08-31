from mmdet.apis import init_detector

from mmcv.runner import build_optimizer, build_runner
from mmdet.models import build_detector

from mmcv import Config

import torch 

# Choose to use a config and initialize the detector
config = 'configs/yolact/yolact_r50_1x8_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/yolact_r50_1x8_coco_20200908-f38d58df.pth'

# initialize the detector
model = init_detector(config, checkpoint, device='cuda:0')

dummy_input["imgs"] = [torch.zeros((1,3,550,550))]
dummy_input["imgs"] = [torch.zeros((1,3,550,550))]

test = model(return_loss=True, rescale=True, **dummy_input)
print(test)

cfg = Config.fromfile(config)
optimizer = build_optimizer(model, cfg.optimizer)

# print(type(optimizer))

