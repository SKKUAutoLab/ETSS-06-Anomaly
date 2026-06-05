import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn.functional as F
device = ("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 512))

    def forward(self, x):
        x = self.resnet(x)
        return x

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0, 0]):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1],self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out, h

class AccidentXai(nn.Module):
    def __init__(self, h_dim, n_layers):
        super(AccidentXai, self).__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.features = FeatureExtractor()
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, n_layers, dropout=[0.5, 0.0])
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self,x,y,toa):
        losses = {'total_loss': 0}
        all_output, all_hidden = [], []
        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(x.device)
        for t in range(x.size(1)):
            x_t = self.features(x[:, t])
            x_t = torch.unsqueeze(x_t, 1)
            output, h = self.gru_net(x_t, h)
            L1 = self._exp_loss(output, y, t, toa=toa, fps=10.0)
            losses['total_loss'] += L1
            all_output.append(output)
        return losses, all_output

    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        target_cls = target[:, 1].to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss