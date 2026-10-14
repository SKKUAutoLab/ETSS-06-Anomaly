# https://github.com/VGraphRNN/VGRNN/blob/master/VGRNN_detection.py
import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch_scatter
from torch_scatter import scatter_add
from torch.autograd import Variable
import torch.nn.functional as F
from src.BayesModels import BayesianLinear
import math

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])
        update_args = [kwargs[arg] for arg in self.update_args]
        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)
        return out

    def message(self, x_j): # pragma: no cover
        return x_j

    def update(self, aggr_out): # pragma: no cover
        return aggr_out

def scatter_(name, src, index, dim_size=None):
    assert name in ['add', 'mean', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0
    out = op(src, index, 0, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]
    if name is 'max':
        out[out == fill_value] = 0
    return out

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def add_self_loops(self, edge_index, edge_weight=None, fill_value=1, num_nodes=None):
        batch_size = edge_index.size(0)
        num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_index = loop_index.unsqueeze(0).repeat(batch_size, 1, 1) # 10 x 2 x 20
        if edge_weight is not None:
            assert edge_weight.size(-1) == edge_index.size(-1)
            loop_weight = edge_weight.new_full((num_nodes,), fill_value)
            loop_weight = loop_weight.unsqueeze(0).repeat(batch_size, 1)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(0), edge_index.size(-1), ), dtype=x.dtype, device=x.device)
        assert edge_weight.size(-1) == edge_index.size(-1)
        # for pytorch 1.4, there are two outputs
        edge_index, edge_weight = self.add_self_loops(edge_index, edge_weight=edge_weight, num_nodes=x.size(1))
        out_batch = []
        for i in range(edge_index.size(0)):
            row, col = edge_index[i]
            deg = scatter_add(edge_weight[i], row, dim=0, dim_size=x.size(1))
            deg_inv = deg.pow(-0.5)
            deg_inv[deg_inv == float('inf')] = 0
            norm = deg_inv[row] * edge_weight[i] * deg_inv[col]
            weight = self.weight.to(x.device)
            x_w = torch.matmul(x[i], weight)
            out = self.propagate('add', edge_index[i], x=x_w, norm=norm)
            out_batch.append(self.act(out))
        out_batch = torch.stack(out_batch)
        return out_batch

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            bias = self.bias.to(aggr_out.device)
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class Graph_GRU_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(Graph_GRU_GCN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h, edge_weight=None):
        h_out = torch.zeros(h.size())
        h_out = h_out.to(inp.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        return h_out

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

class SelfAttAggregate(torch.nn.Module):
    def __init__(self, agg_dim):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))  # (100, 1)
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens, avgsum='sum'):
        maxpool = torch.max(hiddens, dim=1)[0] # (4, 256, 100)
        if avgsum == 'sum':
            avgpool = torch.sum(hiddens, dim=1)
        else:
            avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1) # (4, 512, 100)
        # soft-attention
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial) # (4, 100, 100)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1])) # (4, 100, 512)
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1]) # [4, 100, 1]
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)  # (4, 512, 1)
        return agg_feature.squeeze(dim=-1) # (4, 512)

class BayesianPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, pi=0.5, sigma_1=None, sigma_2=None):
        super(BayesianPredictor, self).__init__()
        self.act = act
        self.l1 = BayesianLinear(input_dim, 64, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l2 = BayesianLinear(64, output_dim, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def forward(self, x, sample=False):
        x = self.act(self.l1(x, sample))
        x = self.l2(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, input, out_dim=2, npass=2, testing=False, eval_uncertain=False):
        npass = npass + 1 if testing else npass
        outputs = torch.zeros(npass, input.size(0), out_dim).to(input.device)
        log_priors = torch.zeros(npass).to(input.device)
        log_variational_posteriors = torch.zeros(npass).to(input.device)
        for i in range(npass):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        if testing:
            outputs[npass] = self(input, sample=False)
        output = outputs.mean(0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        uncertain_alea = torch.zeros(input.size(0), out_dim, out_dim).to(input.device) # [4, 2, 2]
        uncertain_epis = torch.zeros(input.size(0), out_dim, out_dim).to(input.device) # [4, 2, 2]
        if eval_uncertain:
            p = F.softmax(outputs, dim=-1)
            # compute aleatoric uncertainty
            p_diag = torch.diag_embed(p, offset=0, dim1=-2, dim2=-1) # [2, 4, 2, 2]
            p_cov = torch.matmul(p.unsqueeze(-1), p.unsqueeze(-1).permute(0, 1, 3, 2)) # [2, 4, 2, 2]
            uncertain_alea = torch.mean(p_diag - p_cov, dim=0) # [4, 2, 2]
            # compute epistemic uncertainty 
            p_bar = torch.mean(p, dim=0) # [4, 2]
            p_diff_var = torch.matmul((p - p_bar).unsqueeze(-1), (p - p_bar).unsqueeze(-1).permute(0, 1, 3, 2)) # [2, 4, 2, 2]
            uncertain_epis = torch.mean(p_diff_var, dim=0) # [4, 2, 2]
        output_dict = {'pred_mean': output, 'log_prior': log_prior, 'log_posterior': log_variational_posterior, 'aleatoric': uncertain_alea, 'epistemic': uncertain_epis}
        return output_dict

class UString(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True, uncertain_ranking=False):
        super(UString, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.uncertain_ranking = uncertain_ranking
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        # GCN encoder
        self.enc_gcn1 = GCNConv(h_dim + h_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=lambda x: x)
        # RNN layer
        self.rnn = Graph_GRU_GCN(h_dim + h_dim + z_dim, h_dim, n_layers, bias=True)
        # BNN decoder
        self.predictor = BayesianPredictor(n_obj * z_dim, 2)
        if self.with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.n_frames)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, toa, graph, hidden_in=None, edge_weights=None, npass=2, nbatch=80, testing=False, eval_uncertain=False):
        losses = {'cross_entropy': 0, 'log_posterior': 0, 'log_prior': 0, 'total_loss': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
        if self.uncertain_ranking:
            losses.update({'ranking': 0})
            Ut = torch.zeros(x.size(0)).to(x.device)
        all_outputs, all_hidden = [], []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.n_obj, self.h_dim)) # [1, 4, 19, 256]
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)
        for t in range(x.size(1)):
            x_t = self.phi_x(x[:, t]) # [4, 20, 256]
            img_embed = x_t[:, 0, :].unsqueeze(1).repeat(1, self.n_obj, 1).contiguous() # [4, 1, 256]
            obj_embed = x_t[:, 1:, :]  # [4, 19, 256]
            x_t = torch.cat([obj_embed, img_embed], dim=-1) # [4, 19, 512]
            # GCN encoder
            enc = self.enc_gcn1(x_t, graph[:, t], edge_weight=edge_weights[:, t]) # [4, 19, 256]
            z_t = self.enc_gcn2(torch.cat([enc, h[-1]], -1), graph[:, t], edge_weight=edge_weights[:, t]) # [4, 19, 128]
            # BNN decoder
            embed = z_t.view(z_t.size(0), -1) # [10, 19, 128]
            output_dict = self.predictor.sample_elbo(embed, npass=npass, testing=testing, eval_uncertain=eval_uncertain) # [4, 2]
            dec_t = output_dict['pred_mean']
            # recurrence
            h = self.rnn(torch.cat([x_t, z_t], -1), graph[:, t], h, edge_weight=edge_weights[:, t])
            L1 = output_dict['log_posterior'] / nbatch
            L2 = output_dict['log_prior'] / nbatch
            L3 = self._exp_loss(dec_t, y, t, toa=toa, fps=self.fps)
            losses['log_posterior'] += L1
            losses['log_prior'] += L2
            losses['cross_entropy'] += L3
            # uncertainty ranking loss
            if self.uncertain_ranking:
                L5, Ut = self._uncertainty_ranking(output_dict, Ut)
                losses['ranking'] += L5
            all_outputs.append(output_dict)
            all_hidden.append(h[-1])
        if self.with_saa:
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1), 'avg')
            dec = self.predictor_aux(embed_video)
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            losses['auxloss'] = L4
        return losses, all_outputs, all_hidden

    def _exp_loss(self, pred, target, time, toa, fps=20.0):
        # positive example
        target_cls = target[:, 1].to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss

    def _uncertainty_ranking(self, output_dict, Ut, eU_only=True):
        aleatoric = output_dict['aleatoric'] # [4, 2, 2]
        epistemic = output_dict['epistemic'] # [4, 2, 2]
        if eU_only:
            uncertainty = epistemic[:, 0, 0] + epistemic[:, 1, 1]
        else:
            uncertainty = aleatoric[:, 1, 1] + epistemic[:, 1, 1]
        loss = torch.mean(torch.max(torch.zeros_like(Ut).to(Ut.device), uncertainty - Ut))
        return loss, uncertainty