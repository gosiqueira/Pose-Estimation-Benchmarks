# Pose Estimation Benchmarks

![GitHub](https://img.shields.io/github/license/gosiqueira/Pose-Estimation-Benchmarks)

A simple benchmark on current state-of-the-art pose estimation methods.

Current methods:
* AlphaPose (MXNet)
* Alphapose (PyTorch) (comming soon)
* Detectron2 Coco Keypoints
* HRNet: High-Resolution Network
* Higher HRNet (comming soon)
* Simple Baselines for Human Pose Estimation (MXNet)
* Simple Baselines for Human Pose Estimation (PyTorch) (comming soon)

The following results were obtained using high quality videos (1920x1080) with duration between 2 and 10 minutes. Due to nature of the videos, we cannot put them publicly available.

| Method                                              | Repo                                                       | FPS   |
|-----------------------------------------------------|------------------------------------------------------------|-------|
| Alphapose (MXNET)                                   | https://gluon-cv.mxnet.io/model_zoo/pose.html              | 6.68  |
| Alphapose (PyTorch)                                 | https://github.com/MVIG-SJTU/AlphaPose                     | ***   |
| Detectron2 Pose Estimator                           | https://github.com/facebookresearch/detectron2             | 7.96  |
| Higher HRNet                                        | https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation |       |
| HRNet                                               | https://github.com/stefanopini/simple-HRNet                | 11.16 |
| OpenPose                                            | https://github.com/CMU-Perceptual-Computing-Lab/openpose   |       |
| Simple Baseline for Human Pose Estimation (MXNET)   | https://gluon-cv.mxnet.io/model_zoo/pose.html              | 5.28  |
| Simple Baseline for Human Pose Estimation (PyTorch) | https://github.com/microsoft/human-pose-estimation.pytorch |       |