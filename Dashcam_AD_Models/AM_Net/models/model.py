import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, output_cor_dim):
        super(GRUNet, self).__init__()
        self.dropout = [0, 0]
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dense1 = torch.nn.Linear(hidden_dim + output_cor_dim, 256)
        self.dense2 = torch.nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, output_cor):
        out, h = self.gru(x, h)
        out = torch.cat([out, output_cor], dim=-1)
        out = F.dropout(out[:, -1], self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out, h

class CorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(CorGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h

class flow_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(flow_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class SpatialAttention(nn.Module):
    def __init__(self, h_dim):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_dim, 1))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, h_all_in):
        k = []
        v = []
        for key in h_all_in:
            v.append(h_all_in[key])
            k.append(key)
        if len(v) != 0:
            h_in = torch.cat([element for element in v], dim=1) # Eq 6
            m = torch.tanh(h_in)
            alpha = torch.softmax(torch.matmul(m, self.weight), 1) # Eq 7
            roh = torch.mul(h_in, alpha) # Eq 8
            list_roh = []
            for i in range(roh.size(1)):
                list_roh.append(roh[:, i, :].unsqueeze(1).contiguous())
            h_all_in = {}
            for ke, value in zip(k, list_roh):
                h_all_in[ke] = value
        return h_all_in

class AMNet(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100):
        super(AMNet, self).__init__()
        self.x_dim = x_dim # 2048
        self.h_dim = h_dim # 256
        self.n_frames = n_frames # 100
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        # second GRU
        self.n_layers_cor = 1
        self.h_dim_cor = 32
        self.gru_net = GRUNet(h_dim + h_dim, h_dim, 2, self.n_layers, self.h_dim_cor)
        self.weight = torch.Tensor([0.25, 1]).cuda()
        # first GRU
        self.gru_net_cor = CorGRU(4, self.h_dim_cor, self.n_layers_cor)
        self.soft_attention = SpatialAttention(h_dim)
        self.soft_attention_cor = SpatialAttention(self.h_dim_cor)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, x, y, flow):
        losses = {'cross_entropy': 0}
        h_all_in = {}
        h_all_in_cor = {}
        all_outputs = []
        all_labels = []
        for t in range(x.size(1)): # 100 frames
            inp = flow[:, t] # [1, 31, 2048]
            x_val = self.phi_x(inp) # Eq 2: [1, 31, 256]
            img_embed = x_val[:, 0, :].unsqueeze(1) # [1, 1, 256]
            img_embed = img_embed.repeat(1, 30, 1) # [1, 30, 256]
            obj_embed = x_val[:, 1:, :] # [1, 30, 256]
            x_t = torch.cat([obj_embed, img_embed], dim=-1) # [1, 30, 512]
            h_all_out = {}
            h_all_out_cor = {}
            frame_outputs = []
            frame_labels = []
            for bbox in range(30):
                if y[0][t][bbox][0] == 0: # ignore if there is no bounding box
                    continue
                else:
                    track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
                    if track_id in h_all_in:
                        unnormalized_cor = y[0][t][bbox] # [6]
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3]/1080, unnormalized_cor[4]/720]) # normalize bbox
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)
                        h_in_cor = h_all_in_cor[track_id]
                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)
                        h_all_out_cor[track_id] = h_out_cor
                        h_in = h_all_in[track_id]
                        x_obj = x_t[0][bbox]
                        x_obj = torch.unsqueeze(x_obj, 0)
                        x_obj = torch.unsqueeze(x_obj, 0)
                        output, h_out = self.gru_net(x_obj, h_in, output_cor)
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        # calc loss
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out
                    else: # If object was not found in the previous frame
                        unnormalized_cor = y[0][t][bbox] # [6]
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3]/1080, unnormalized_cor[4]/720]) # normalize bbox
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)
                        h_in_cor = Variable(torch.zeros(self.n_layers_cor, x.size(0),  self.h_dim_cor))
                        h_in_cor = h_in_cor.to(x.device)
                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)
                        h_in = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
                        h_in = h_in.to(x.device)
                        x_obj = x_t[0][bbox]
                        x_obj = torch.unsqueeze(x_obj, 0)
                        x_obj = torch.unsqueeze(x_obj, 0)
                        output, h_out = self.gru_net(x_obj, h_in, output_cor)
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        # calc loss
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out
                        h_all_out_cor[track_id] = h_out_cor
            all_outputs.append(frame_outputs)
            all_labels.append(frame_labels)
            # h_all_in = {}
            h_all_in = h_all_out.copy()
            h_all_in = self.soft_attention(h_all_in)
            # h_all_in_cor = {}
            h_all_in_cor = h_all_out_cor.copy()
            h_all_in_cor = self.soft_attention_cor(h_all_in_cor)
        return losses, all_outputs, all_labels