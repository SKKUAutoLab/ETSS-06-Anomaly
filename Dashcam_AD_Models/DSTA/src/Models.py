import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=[0, 0]):
        super(AccidentPredictor, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        x = self.dense2(x)
        return x

class frame_AttAggregate(torch.nn.Module): # DTA module
    def __init__(self, agg_dim):
        super(frame_AttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(512, 512))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens):
        m = torch.tanh(hiddens)
        alpha = torch.softmax(torch.matmul(m, self.weight), 0) # Eq 5
        roh = torch.mul(hiddens, alpha) # Eq 6
        new_h = torch.sum(roh, 0)
        return new_h

class SelfAttAggregate(torch.nn.Module): # TSAA module
    def __init__(self, agg_dim):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens):
        hiddens = hiddens.unsqueeze(0)
        hiddens = hiddens.permute(1, 0, 2, 3)
        maxpool = torch.max(hiddens, dim=1)[0]
        avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)
        return agg_feature.squeeze(dim=-1)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0, 0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1], self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        return out, h

class SpatialAttention(torch.nn.Module): # DSA module
    def __init__(self, h_dim, z_dim, n_layers, n_obj):
        super(SpatialAttention, self).__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_obj = n_obj
        self.batch_size = 10
        self.weights_att_w = Variable(torch.empty(h_dim, n_layers).normal_(mean=0.0, std=0.5))
        self.weights_att_ua = Variable(torch.empty(h_dim, h_dim).normal_(mean=0.0, std=0.01))
        self.weights_att_ba = Variable(torch.zeros(h_dim))
        self.weights_att_wa = Variable(torch.empty(h_dim, h_dim).normal_(mean=0.0, std=0.01))

    def forward(self, obj_embed, h, zeros_object):
        self.weights_att_ua = self.weights_att_ua.to(h.device)
        self.weights_att_ba = self.weights_att_ba.to(h.device)
        self.weights_att_wa = self.weights_att_wa.to(h.device)
        self.weights_att_w = self.weights_att_w.to(h.device)
        brcst_w = self.weights_att_w.unsqueeze(0).repeat(self.n_obj, 1, 1)
        obj_embed = obj_embed.permute(1, 0, 2)
        image_part = torch.matmul(obj_embed, self.weights_att_ua.unsqueeze(0).repeat(self.n_obj, 1, 1)) + self.weights_att_ba
        h = h.permute(1, 0, 2)
        e = torch.tanh(torch.matmul(h, self.weights_att_wa).permute(1, 0, 2) + image_part)
        alphas = torch.mul(torch.softmax(torch.sum(torch.matmul(e, brcst_w),2 ), 0), zeros_object)  # Eq 2
        al_alphas = alphas
        obj_embed = torch.mul(alphas.unsqueeze(2),obj_embed)
        obj_embed = torch.sum(obj_embed, 0)
        obj_embed = obj_embed.unsqueeze(0)
        obj_embed = obj_embed.permute(1, 0, 2)
        return obj_embed, al_alphas

class DSTA(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True):
        super(DSTA, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.sp_attention = SpatialAttention(h_dim,z_dim, n_layers, n_obj)
        self.gru_net = GRUNet(h_dim + h_dim, h_dim, 2, n_layers, dropout=[0.5, 0.0])
        self.frame_aggregation = frame_AttAggregate(5)
        if self.with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.n_frames)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, toa, hidden_in=None):
        losses = {'cross_entropy': 0, 'total_loss': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
        all_outputs, all_hidden = [], []
        all_alphas = []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)
        zeros_object_1 = torch.sum(x[:, :, 1:self.n_obj + 1, :].permute(1, 2, 0, 3), 3)
        zeros_object_2 = ~zeros_object_1.eq(0)
        zeros_object = zeros_object_2.float()
        h_list = []
        for t in range(x.size(1)):
            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1)
            obj_embed = x_t[:, 1:, :]
            obj_embed, alphas = self.sp_attention(obj_embed, h, zeros_object[t])
            x_t = torch.cat([obj_embed, img_embed], dim=-1)
            h_list.append(h)
            all_alphas.append(alphas)
            if t == 2:
                h_staked = torch.stack((h_list[t], h_list[t-1], h_list[t-2]), dim=0)
                h = self.frame_aggregation(h_staked)
            elif t == 3:
                h_staked = torch.stack((h_list[t], h_list[t-1], h_list[t-2], h_list[t-3]), dim=0)
                h = self.frame_aggregation(h_staked)
            elif t > 3:
                h_staked = torch.stack((h_list[t], h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4]), dim=0)
                h = self.frame_aggregation(h_staked)
            # Uncomment below for DAD dataset
            # elif t==4:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            # elif t==5:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            # elif t==6:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            # elif t==7:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            # elif t==8:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7], h_list[t-8]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            # elif t>8:
            #     h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7], h_list[t-8], h_list[t-9]),dim=0)
            #     h = self.frame_aggregation(h_staked)
            output, h = self.gru_net(x_t, h)
            L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])
        if self.with_saa:
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            losses['auxloss'] = L4
        return losses, all_outputs, all_hidden, all_alphas

    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        target_cls = target[:, 1].to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss