import torch
import torch.nn as nn
import torch.nn.init as torch_init
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, 'weight'):  # Check if module has weight attribute
            torch_init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, feature_dim, seq_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, feature_dim, seq_len))

    def forward(self, x):
        # Expand the positional encoding to match the batch size
        batch_size = x.size(0)
        pos_enc = self.positional_encoding.expand(batch_size, -1, -1)
        return x + pos_enc


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.positional_encoder = LearnedPositionalEncoding(feature_dim=in_channels, seq_len=50)

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        bn = nn.BatchNorm1d
        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        x = self.positional_encoder(x)
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
       
        W_y = self.W(y)
        
        z = W_y #+ x

        if return_nl_map:
            return z, f_div_C
        return z


#############################################################
# DIFFERENT TEMPORAL MODELING APPROACHES
#############################################################

class DTCA(nn.Module):
    """Original Dilated Temporal Conv-Attention (DTCA) Module"""
    def __init__(self, len_feature):
        super(DTCA, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=768, kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=768, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=768, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=768, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=int(len_feature/2), out_channels=768, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3),
        )
        self.conv_6 = nn.Sequential(
            nn.Conv1d(in_channels=int(len_feature/2), out_channels=768, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            bn(768),
            nn.Dropout(0.3),
        )
        self.conv_7 = nn.Sequential(
            nn.Conv1d(in_channels=768*6, out_channels=768, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3)
        )

        self.non_local1 = NonLocalBlockND(768, bn_layer=True)
        self.non_local2 = NonLocalBlockND(768, bn_layer=True)
        self.non_local3 = NonLocalBlockND(768, bn_layer=True)

    def forward(self, x):
        # x: (B, T, F)
        out_fh = x[:,:,:int(768/2)].permute(0, 2, 1)
        out_sh = x[:,:,int(768/2):].permute(0, 2, 1)
        out = x.permute(0, 2, 1)
        
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)

        out4 = self.conv_4(out)
        out4 = self.non_local1(out4)
        
        out5 = self.conv_5(out_fh)
        out5 = self.non_local2(out5)
        out6 = self.conv_6(out_sh)
        out6 = self.non_local3(out6)

        out = torch.cat((out1, out2, out3, out4, out5, out6), dim=1)
        out = self.conv_7(out)   # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)

        return out


class TCN(nn.Module):
    """Temporal Convolutional Network Module"""
    def __init__(self, len_feature):
        super(TCN, self).__init__()
        self.len_feature = len_feature
        
        # TCN layers with residual connections
        self.tcn_layers = nn.ModuleList([
            self._make_tcn_layer(len_feature, 768, 3, 1, 1),
            self._make_tcn_layer(768, 768, 3, 2, 2),
            self._make_tcn_layer(768, 768, 3, 4, 4)
        ])
        
        # Final layer to match output dimensions
        self.final_layer = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3)
        )
    
    def _make_tcn_layer(self, in_channels, out_channels, kernel_size, dilation, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        
        residual = x if self.len_feature == 768 else nn.Conv1d(self.len_feature, 768, 1)(x)
        
        # Apply TCN layers
        out = self.tcn_layers[0](x)
        for layer in self.tcn_layers[1:]:
            out = layer(out) + out  # Residual connection within the TCN
        
        out = self.final_layer(out)
        out = out + residual  # Global residual connection
        
        return out.permute(0, 2, 1)  # (B, T, 768)


class Transformer(nn.Module):
    """Transformer-based Temporal Module"""
    def __init__(self, len_feature):
        super(Transformer, self).__init__()
        self.len_feature = len_feature
        
        # Initial projection if input feature dimension doesn't match model dimension
        self.input_projection = nn.Conv1d(len_feature, 768, kernel_size=1) if len_feature != 768 else nn.Identity()
        
        # Positional encoding
        self.positional_encoding = LearnedPositionalEncoding(feature_dim=768, seq_len=50)
        
        # Multi-head attention layers
        self.mha_layers = nn.ModuleList([
            self._make_mha_block() for _ in range(3)
        ])
        
        # Final layer to match output dimensions
        self.final_layer = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3)
        )
    
    def _make_mha_block(self):
        return nn.Sequential(
            nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True),
            nn.Dropout(0.3),
            nn.LayerNorm(768)
        )
    
    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = x.permute(0, 2, 1)  # (B, T, 768)
        
        # Apply transformer layers
        for mha_block in self.mha_layers:
            mha, dropout, ln = mha_block
            attn_output, _ = mha(x, x, x)
            x = x + dropout(attn_output)  # Residual connection
            x = ln(x)
        
        # Final projection
        x = x.permute(0, 2, 1)  # (B, 768, T)
        x = self.final_layer(x)
        x = x.permute(0, 2, 1)  # (B, T, 768)
        
        return x


class LSTM(nn.Module):
    """Bidirectional LSTM-based Temporal Module"""
    def __init__(self, len_feature):
        super(LSTM, self).__init__()
        self.len_feature = len_feature
        
        # Initial projection if needed
        self.input_projection = nn.Linear(len_feature, 512) if len_feature != 512 else nn.Identity()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(384*2, 768),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x: (B, T, F)
        x = self.input_projection(x)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project to output dimension
        out = self.output_projection(lstm_out)
        
        return out


class ConvLSTM(nn.Module):
    """Convolutional LSTM Module - combining TCN and LSTM"""
    def __init__(self, len_feature):
        super(ConvLSTM, self).__init__()
        self.len_feature = len_feature
        
        # Convolutional part (similar to TCN but simplified)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # LSTM part
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(384*2, 768),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv_layers(x)  # (B, 512, T)
        x = x.permute(0, 2, 1)  # (B, T, 512)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project to output dimension
        out = self.output_projection(lstm_out)
        
        return out


#############################################################
# AGGREGATION MODULE WITH SWITCH FOR TEMPORAL MODELS
#############################################################

class Aggregate(nn.Module):
    def __init__(self, len_feature, temporal_model):
        """
        Aggregate module with a switch for different temporal models.
        
        Args:
            len_feature: Feature dimension
            temporal_model: Which temporal model to use, options:
                'dtca' - Dilated Temporal Conv-Attention (original)
                'tcn' - Temporal Convolutional Network
                'transformer' - Transformer-based
                'lstm' - LSTM-based
                'convlstm' - Convolutional LSTM
        """
        super(Aggregate, self).__init__()
        self.len_feature = len_feature
        self.temporal_model = temporal_model
        
        # Initialize the selected temporal model
        if temporal_model == 'dtca':
            self.model = DTCA(len_feature)
        elif temporal_model == 'tcn':
            self.model = TCN(len_feature)
        elif temporal_model == 'transformer':
            self.model = Transformer(len_feature)
        elif temporal_model == 'lstm':
            self.model = LSTM(len_feature)
        elif temporal_model == 'convlstm':
            self.model = ConvLSTM(len_feature)
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")
            
    def forward(self, x):
        return self.model(x)


#############################################################
# MAIN MODEL WITH TEMPORAL MODELING SWITCH
#############################################################

class Model(nn.Module):
    # =====================================================================
    # SWITCH IS HERE - Set 'temporal_model' parameter when creating Model
    # Options: 'dtca' (default), 'tcn', 'transformer', 'lstm', 'convlstm'
    # Example: model = Model(n_features=768, batch_size=32, temporal_model='tcn')
    # =====================================================================
    def __init__(self, n_features, batch_size, temporal_model='convlstm'):
        """
        Main model with a switch to select different temporal models.
        
        Args:
            n_features: Feature dimension
            batch_size: Batch size
            temporal_model: Which temporal model to use, options:
                'dtca' - Dilated Temporal Conv-Attention (original)
                'tcn' - Temporal Convolutional Network
                'transformer' - Transformer-based
                'lstm' - LSTM-based
                'convlstm' - Convolutional LSTM
        """
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 40
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10
        
        # Pass the temporal model selection to the Aggregate module
        self.Aggregate = Aggregate(len_feature=n_features, temporal_model=temporal_model)
        
        # Rest of the model architecture remains the same
        self.fc1 = nn.Linear(768, 512)  # Note: All temporal models output 768 dim features
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        out = self.Aggregate(out)
        
        out = self.drop_out(out)
        
        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)

        normal_features = features[0:self.batch_size*10]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*10:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes).cuda()

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1, largest=False)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        
        abnormal_features = abnormal_features.view(n_size, ncrops, t, 768)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0)
        total_select_abn_feature = total_select_abn_feature.to(abnormal_features.device)

        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            feat_select_abn = feat_select_abn.to(total_select_abn_feature.device)
            
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out2(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1, largest=False)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, 768)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        total_select_nor_feature = total_select_nor_feature.to(normal_features.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, feat_magnitudes