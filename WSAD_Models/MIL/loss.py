import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0, model=None):
    loss = torch.tensor(0.0).cuda()
    sparsity = torch.tensor(0.0).cuda()
    smooth = torch.tensor(0.0).cuda()
    lambda1, lambda2, lambda3 = 0.00008, 0.00008, 0.01
    model_loss = 0.0
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)
    for i in range(batch_size): # bs = 30
        normal_index = torch.randperm(30).cuda()
        anomaly_index = torch.randperm(30).cuda()
        y_normal = y_pred[i, 32:][normal_index]
        y_anomaly = y_pred[i, :32][anomaly_index]
        loss += F.relu(1.0 - torch.max(y_anomaly) + torch.max(y_normal))
        sparsity += torch.sum(y_anomaly) * lambda2
        smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32])**2) * lambda1
    for param in model.parameters():
        model_loss += torch.norm(param)
    loss = (loss + sparsity + smooth + lambda3 * model_loss) / batch_size
    return loss