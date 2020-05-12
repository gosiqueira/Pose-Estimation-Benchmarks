import mxnet as mx
import numpy as np

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord_alpha_pose

from methods.BaseEstimator import BaseEstimator


class SimpleBaselinesMXNet(BaseEstimator):
    def __init__(self):
        self.ctx = self.try_gpu()
        self.detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
        self.pose_net = model_zoo.get_model('simple_pose_resnet50_v1b', pretrained=True)

        self.detector.reset_class(['person'], reuse_weights=['person'])
        self.transformer = data.transforms.presets.yolo.transform_test
        
    def get_poses(self, image):
        x, image = self.transformer(mx.nd.array(image), short=512)
        x.as_in_context(self.ctx)

        class_IDs, scores, bounding_boxs = self.detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(image, class_IDs, scores, bounding_boxs)

        if pose_input is not None:
            predicted_heatmap = self.pose_net(pose_input)
            return heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
        else:
            return mx.nd.array([]), mx.nd.array([])

    def eval(self, true, pred):
        raise NotImplementedError

    def try_gpu(self):
        if mx.context.num_gpus() > 1:
            return mx.gpu()
        else:
            print('Warning: You are running using cpu. The running time is lower using this context.')
            return mx.cpu()