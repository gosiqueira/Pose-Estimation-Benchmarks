from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader

from methods.BaseEstimator import BaseEstimator


class AlphaPosePyTorch(BaseEstimator):
    def __init__(self):
        self.device = try_gpu()
        cfg = update_config(args.cfg)
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        pose_model.load_state_dict(torch.load(args.checkpoint, map_location=self.device))

        pose_model.to(self.device)

    def get_poses(self, image):
        with torch.no_grad():
        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
        if orig_img is None:
            break

        inps = inps.to(args.device)
        heatmap = pose_model(inps)

        print(heatmap)
        return heatmap

    def eval(self, true, pred):
        raise NotImplementedError

    def try_gpy(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print('Warning: You are running using cpu. The running time is lower using this device.')
            return torch.device('cpu')