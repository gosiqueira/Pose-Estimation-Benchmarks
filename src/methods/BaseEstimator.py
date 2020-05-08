from abc import ABCMeta, abstractmethod

class BaseEstimator(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_poses(self, image):
        pass

    @abstractmethod
    def eval(self, true, pred):
        pass