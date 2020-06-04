import torch

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader

from methods.BaseEstimator import BaseEstimator


class AlphaPosePyTorch(BaseEstimator):
    def __init__(self):
        self.device = try_gpu()
        self.cfg = update_config('configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')

        self.detector = get_detector({'detector': yolo})
        self.detector.load_model()

        self.pose_net = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_net.load_state_dict(torch.load('pretrained_models/fast_res50_256x192.pth', map_location=self.device))

        pose_model.to(self.device)

    def get_poses(self, image):
        with torch.no_grad():
            x = self.detector.image_preprocess(image)
            x = x.to(self.device)
            inps = self.detector.model(x)
            
            inps = inps.to(self.device)
            heatmap = self.pose_net(inps)

            print(heatmap)
            return heatmap

    def eval(self, true, pred):
        raise NotImplementedError

    def try_gpu(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print('Warning: You are running using cpu. The running time is lower using this device.')
            return torch.device('cpu')