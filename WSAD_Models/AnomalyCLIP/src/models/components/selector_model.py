import torch
from torch import nn

class SelectorModel(nn.Module):
    def __init__(self, classnames: list, normal_id: int, logit_scale: nn.Parameter, num_segments: int, seg_length: int, select_idx_dropout_topk: float,
                 select_idx_dropout_bottomk: float, num_topk: int, num_bottomk: int):
        super().__init__()
        self.classnames = classnames
        self.normal_id = normal_id
        self.logit_scale = logit_scale
        self.num_segments = num_segments
        self.seg_length = seg_length
        self.select_idx_dropout_topk = select_idx_dropout_topk
        self.select_idx_dropout_bottomk = select_idx_dropout_bottomk
        self.num_topk = num_topk
        self.num_bottomk = num_bottomk
        self.bn_layer = nn.BatchNorm1d(len(classnames) - 1, affine=False)

    def forward(self, image_features, text_features, labels, ncentroid, test_mode):
        image_features = torch.reshape(image_features, (-1, image_features.shape[-1]))
        text_features_except_normal = torch.cat((text_features[: self.normal_id], text_features[(self.normal_id + 1) :]), dim=0)
        text_features = text_features_except_normal - ncentroid
        image_features = image_features - ncentroid
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        logits = self.bn_layer(logits)
        logits = logits.view(-1, logits.shape[-1])
        if test_mode:
            return logits
        else:
            logits = logits.view(-1, self.num_segments * self.seg_length, logits.shape[-1])
            topk_mask, bottomk_mask = self.generate_mask(logits)
            topk_mask = topk_mask.to(image_features.device)
            bottomk_mask = bottomk_mask.to(image_features.device)
            logits_topk, idx_topk = self.select_topk(logits, labels, topk_mask)
            idx_topk_abn, idx_topk_nor = (idx_topk[: idx_topk.shape[0] // 2], idx_topk[idx_topk.shape[0] // 2 :])
            logits_bottomk, idx_bottomk = self.select_bottomk(logits, labels, bottomk_mask)
            idx_bottomk_abn = idx_bottomk[: idx_bottomk.shape[0] // 2]
            logits = logits.view(-1, logits.shape[-1])
            logits_topk = logits_topk.view(-1, logits_topk.shape[-1])
            logits_bottomk = logits_bottomk.view(-1, logits_bottomk.shape[-1])
            return logits, logits_topk, logits_bottomk, idx_topk_abn, idx_topk_nor, idx_bottomk_abn

    def generate_mask(self, logits):
        select_idx = torch.ones((logits.shape[0], self.num_segments))
        topk_select_idx = select_idx * (1 - self.select_idx_dropout_topk)
        bottomk_select_idx = select_idx * (1 - self.select_idx_dropout_bottomk)
        topk_mask = torch.bernoulli(topk_select_idx).unsqueeze(2).expand([-1, -1, logits.shape[-1]])
        bottomk_mask = torch.bernoulli(bottomk_select_idx).unsqueeze(2).expand([-1, -1, logits.shape[-1]])
        if self.select_idx_dropout_topk == self.select_idx_dropout_bottomk:
            topk_mask = bottomk_mask
        return topk_mask, bottomk_mask

    def select_topk_idx(self, logits, labels, mask):
        b, t, num_classes = logits.shape
        logits_sum = logits.view(-1, self.num_segments, self.seg_length, num_classes)
        logits_sum = torch.sum(logits_sum, dim=2)
        min_value = -1e6
        logits_drop = torch.where(mask == 0, torch.ones_like(logits_sum) * min_value, logits_sum)
        alogits_drop = logits_drop[: b // 2]
        alabels = labels[: b // 2]
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)
        idx_topk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            idx_topk_abn_i = torch.topk(alogits_i[:, alabels_i], self.num_topk, dim=0, largest=True)[1]
            idx_topk_abn_i = idx_topk_abn_i.unsqueeze(0)
            idx_topk_abn.append(idx_topk_abn_i)
        idx_topk_abn = torch.cat(idx_topk_abn)
        nlogits_drop = logits_drop[logits_drop.shape[0] // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)
        idx_topk_nor = torch.topk(nlogits_drop, k=self.num_topk, dim=1, largest=True)[1]
        return idx_topk_abn, idx_topk_nor

    def select_topk(self, logits, labels, mask):
        b, t, num_classes = logits.shape
        idx_topk_abn, idx_topk_nor = self.select_topk_idx(logits, labels, mask)
        idx_topk_abn_logits = idx_topk_abn.unsqueeze(2).expand([-1, -1, num_classes])
        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(-1, self.num_segments, self.seg_length, num_classes)
        idx_topk_abn_logits = idx_topk_abn_logits.unsqueeze(2).expand([-1, -1, self.seg_length, -1])
        total_select_abn_logits = []
        for a_logit, a_topk_idx in zip(alogits, idx_topk_abn_logits):
            logit_topk_abn = torch.gather(a_logit, 0, a_topk_idx)
            total_select_abn_logits.append(logit_topk_abn)
        total_select_abn_logits = torch.cat(total_select_abn_logits)
        idx_topk_nor_logits = idx_topk_nor.unsqueeze(2).expand([-1, -1, num_classes])
        nlogits = logits[logits.shape[0] // 2 :]
        nlogits = nlogits.view(-1, self.num_segments, self.seg_length, num_classes)
        idx_topk_nor_logits = idx_topk_nor_logits.unsqueeze(2).expand([-1, -1, self.seg_length, -1])
        total_select_nor_logits = []
        for n_logit, n_topk_idx in zip(nlogits, idx_topk_nor_logits):
            logit_topk_nor = torch.gather(n_logit, 0, n_topk_idx)
            total_select_nor_logits.append(logit_topk_nor)
        total_select_nor_logits = torch.cat(total_select_nor_logits)
        total_select_logits = torch.cat((total_select_abn_logits, total_select_nor_logits))
        idx_logits = torch.cat((idx_topk_abn, idx_topk_nor), dim=0)
        return total_select_logits, idx_logits

    def select_bottomk_idx(self, logits, labels, mask):
        b, t, num_classes = logits.shape
        logits_sum = logits.view(-1, self.num_segments, self.seg_length, num_classes)
        logits_sum = torch.sum(logits_sum, dim=2)
        max_value = 1e6
        logits_drop = torch.where(mask == 0, torch.ones_like(logits_sum) * max_value, logits_sum)
        alogits_drop = logits_drop[: b // 2]
        alabels = labels[: b // 2]
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)
        idx_bottomk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            idx_bottomk_abn_i = torch.topk(alogits_i[:, alabels_i], self.num_bottomk, dim=0, largest=False)[1]
            idx_bottomk_abn_i = idx_bottomk_abn_i.unsqueeze(0)
            idx_bottomk_abn.append(idx_bottomk_abn_i)
        idx_bottomk_abn = torch.cat(idx_bottomk_abn)
        nlogits_drop = logits_drop[b // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)
        idx_bottomk_nor = torch.topk(nlogits_drop, k=self.num_bottomk, dim=1, largest=False)[1]
        return idx_bottomk_abn, idx_bottomk_nor

    def select_bottomk(self, logits, labels, mask):
        b, t, num_classes = logits.shape
        idx_bottomk_abn, idx_bottomk_nor = self.select_bottomk_idx(logits, labels, mask)
        idx_bottomk_abn_logits = idx_bottomk_abn.unsqueeze(2).expand([-1, -1, num_classes])
        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(-1, self.num_segments, self.seg_length, num_classes)
        idx_bottomk_abn_logits = idx_bottomk_abn_logits.unsqueeze(2).expand([-1, -1, self.seg_length, -1])
        total_select_abn_logits = []
        for a_logit, a_bottomk_idx in zip(alogits, idx_bottomk_abn_logits):
            logit_bottomk_abn = torch.gather(a_logit, 0, a_bottomk_idx)
            total_select_abn_logits.append(logit_bottomk_abn)
        total_select_abn_logits = torch.cat(total_select_abn_logits)
        idx_bottomk_nor_logits = idx_bottomk_nor.unsqueeze(2).expand([-1, -1, num_classes])
        nlogits = logits[logits.shape[0] // 2 :]
        nlogits = nlogits.view(-1, self.num_segments, self.seg_length, num_classes)
        idx_bottomk_nor_logits = idx_bottomk_nor_logits.unsqueeze(2).expand([-1, -1, self.seg_length, -1])
        total_select_nor_logits = []
        for n_logit, n_bottomk_idx in zip(nlogits, idx_bottomk_nor_logits):
            logit_bottomk_nor = torch.gather(n_logit, 0, n_bottomk_idx)
            total_select_nor_logits.append(logit_bottomk_nor)
        total_select_nor_logits = torch.cat(total_select_nor_logits)
        total_select_logits = torch.cat((total_select_abn_logits, total_select_nor_logits))
        idx_logits = torch.cat((idx_bottomk_abn, idx_bottomk_nor), dim=0)
        return total_select_logits, idx_logits