import torch
import torch.nn as nn
import torch.nn.functional as F

learning_rate = 1e-3
n_img_hidden = 256
n_hidden = 512
hidden = 256
n_att_hidden = 256


def normal_param(*shape):
    return nn.Parameter(torch.empty(*shape).normal_(mean=0.0, std=0.01))


class PeepholeLSTMCell(nn.Module):
    """LSTM cell with peephole connections, matching
    tf.contrib.rnn.LSTMCell(use_peepholes=True, state_is_tuple=False).

    The state is the concatenation [c, h] of shape (batch, 2 * num_units).
    """

    def __init__(self, num_inputs, num_units, forget_bias=1.0):
        super(PeepholeLSTMCell, self).__init__()
        self.num_units = num_units
        self.forget_bias = forget_bias

        self.state_size = 2 * num_units

        # kernel: (num_inputs + num_units, 4 * num_units) for gates [i, j, f, o]
        self.kernel = normal_param(num_inputs + num_units, 4 * num_units)
        self.bias = nn.Parameter(torch.zeros(4 * num_units))

        # peephole (diagonal) connections
        self.w_i_diag = normal_param(num_units)
        self.w_f_diag = normal_param(num_units)
        self.w_o_diag = normal_param(num_units)

    def forward(self, inputs, state):
        c_prev = state[:, 0:self.num_units]
        h_prev = state[:, self.num_units:]

        lstm_matrix = torch.matmul(torch.cat([inputs, h_prev], 1), self.kernel) + self.bias
        i, j, f, o = torch.split(lstm_matrix, self.num_units, dim=1)

        i = torch.sigmoid(i + self.w_i_diag * c_prev)
        f = torch.sigmoid(f + self.forget_bias + self.w_f_diag * c_prev)
        c = f * c_prev + i * torch.tanh(j)
        o = torch.sigmoid(o + self.w_o_diag * c)
        h = o * torch.tanh(c)

        new_state = torch.cat([c, h], 1)
        return h, new_state


class LSTM_RNN(nn.Module):

    def __init__(self, n_input, n_steps, n_objects, batch_size, n_classes):
        super(LSTM_RNN, self).__init__()

        self.n_input = n_input
        self.n_steps = n_steps
        self.n_objects = n_objects
        self.batch_size = batch_size
        self.n_classes = n_classes

        # Graph weights
        self.weights = nn.ParameterDict({
            'em_img': normal_param(n_input, n_img_hidden),  # Hidden layer weights
            'em_obj': normal_param(n_input, n_att_hidden),
            'att_wa': normal_param(n_hidden, n_att_hidden),
            'Spatial_W_g': normal_param(hidden, hidden),
            'Spatial_W_theta': normal_param(hidden, hidden),
            'Spatial_W_phi': normal_param(hidden, hidden),
            'out': normal_param(n_hidden, n_classes),
        })
        self.biases = nn.ParameterDict({
            'em_img': normal_param(n_img_hidden),
            'em_obj': normal_param(n_att_hidden),
            'Spatial_W_g': normal_param(hidden),
            'Spatial_W_theta': normal_param(hidden),
            'Spatial_W_phi': normal_param(hidden),
            'out': normal_param(n_classes),
        })

        self.lstm_cell = PeepholeLSTMCell(n_img_hidden + n_att_hidden, n_hidden)

    def forward(self, x, y, keep):
        """
        :param x: (batch_size, n_steps, n_objects, 4096) float tensor
        :param y: (batch_size, n_classes) one-hot float tensor
        :param keep: dropout probability applied to the LSTM output
                     (0.5 during training, 0.0 during testing)
        :return: loss, soft_pred (batch_size, n_steps), zt
        """
        n_steps = self.n_steps
        n_objects = self.n_objects
        n_input = self.n_input
        batch_size = x.shape[0]
        weights = self.weights
        biases = self.biases

        # mask of frames whose object features are all zero:
        # (n_steps, n_objects - 1, batch_size)
        zeros_object = (x[:, :, 1:n_objects, :].permute(1, 2, 0, 3).sum(3) != 0).float()

        # init LSTM parameters
        istate = x.new_zeros(batch_size, self.lstm_cell.state_size)
        h_prev = x.new_zeros(batch_size, n_hidden)
        loss = 0.0
        soft_pred = []

        for t in range(n_steps):

            X = x[:, t, :, :].permute(1, 0, 2)

            full_frame = X[0, :, :]

            # linear embedding of full frame features
            image = torch.matmul(full_frame, weights['em_img']) + biases['em_img']
            x1 = X[1:n_objects, :, :]

            x2 = x1.reshape(-1, n_input)

            # linear embedding of object features
            n_object = torch.matmul(x2, weights['em_obj']) + biases['em_obj']

            n_object = n_object.reshape(n_objects - 1, batch_size, n_att_hidden)
            n_object = n_object * zeros_object[t].unsqueeze(2)

            image_part1 = torch.matmul(n_object, weights['Spatial_W_theta']) + biases['Spatial_W_theta']

            k = torch.matmul(h_prev, weights['att_wa'])
            theta = torch.tanh(k + image_part1)

            image_part2 = torch.matmul(n_object, weights['Spatial_W_phi']) + biases['Spatial_W_phi']

            phi = torch.tanh(torch.matmul(h_prev, weights['att_wa']) + image_part2)

            g = torch.matmul(n_object, weights['Spatial_W_g']) + biases['Spatial_W_g']

            theta = theta.permute(1, 0, 2)
            phi = phi.permute(1, 2, 0)

            c = F.softmax(torch.matmul(theta, phi), dim=2)
            g = g.permute(1, 0, 2)

            vector = torch.matmul(c, g)
            vector = vector.permute(1, 0, 2)

            final = torch.sum(vector + n_object, 0)
            fusion = torch.cat([image, final], 1)

            outputs, istate = self.lstm_cell(fusion, istate)
            outputs = F.dropout(outputs, p=keep, training=(keep > 0))

            h_prev = outputs

            zt = torch.matmul(outputs, weights['out']) + biases['out']  # b x n_classes
            # save the predict of each time step

            soft_pred.append(F.softmax(zt, dim=1)[:, 1].reshape(batch_size, 1))

            cross_entropy = -torch.sum(y * F.log_softmax(zt, dim=1), dim=1)

            pos_loss = -torch.exp(torch.tensor(-(n_steps - t - 1) / 20.0, device=x.device)) * (-cross_entropy)

            neg_loss = cross_entropy

            temp_loss = torch.mean(pos_loss * y[:, 1] + neg_loss * y[:, 0])
            loss = loss + temp_loss

        soft_pred = torch.cat(soft_pred, 1)

        return loss, soft_pred, zt
