import torch, torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class CosineSimilarityLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        # empty_label=17,
        # class_weights=[1.0, 1.0],
        input_dict=None
    ):
        
        super().__init__()

        self.weight = weight
        if input_dict is None:
            self.input_dict = {
                'T_orig_features': 'T_orig_features',
                'T_proj_features': 'T_proj_features'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_similarity

        # self.empty_label = empty_label
        # self.class_weights = torch.tensor(class_weights)
        # self.class_weights = 2 * F.normalize(self.class_weights, 1, -1)
        # print(self.__class__, self.class_weights)

    def loss_similarity(self, T_orig_features, T_proj_features):

        B, C, _ = T_orig_features.shape
        T_orig_features = T_orig_features.view(B*C, -1)
        T_orig_norm = F.normalize(T_orig_features, p=2, dim=1)
        S_orig = torch.matmul(T_orig_norm, T_orig_norm.t())
        
        tot_loss = 0.
        for idx, t_proj_feature in enumerate(T_proj_features):
                    
            t_proj_feature = t_proj_feature.view(B*C, -1)

            T_proj_norm = F.normalize(t_proj_feature, p=2, dim=1)

            S_proj = torch.matmul(T_proj_norm, T_proj_norm.t())

            loss = F.mse_loss(S_proj, S_orig.detach())

            tot_loss += loss

        return tot_loss / len(T_proj_features)
        # T_orig_features = T_orig_features.view(B*C, -1)
        # T_proj_features = T_proj_features.view(B*C, -1)

        # T_orig_norm = F.normalize(T_orig_features, p=2, dim=1)
        # T_proj_norm = F.normalize(T_proj_features, p=2, dim=1)

        # S_orig = torch.matmul(T_orig_norm, T_orig_norm.t())
        # S_proj = torch.matmul(T_proj_norm, T_proj_norm.t())

        # loss = F.mse_loss(S_proj, S_orig.detach())
        # return loss