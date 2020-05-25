import torch
from SimpleHRNet import SimpleHRNet

from methods.BaseEstimator import BaseEstimator


class HRNet(BaseEstimator):
    def __init__(self):
        self.device = self.try_gpu()
        self.model = SimpleHRNet(48, 17, './methods/hrnet/weights/pose_hrnet_w48_384x288.pth', device=self.device)

    def get_poses(self, image):
        return self.model.predict(image)

    def eval(self, true, pred):
        raise NotImplementedError

    def try_gpu(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print('Warning: You are running using cpu. The running time is lower using this device.')
            return torch.device('cpu')