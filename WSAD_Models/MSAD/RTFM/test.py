from torch.utils.data import DataLoader
import torch
from model import Model
from dataset import Dataset
from test_10crop import test
import option
from config import *

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    test_loader = DataLoader(Dataset(args, test_mode=True), batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    checkpoint = torch.load(args.testing_model)
    model = Model(args.feature_size, args.batch_size)  
    model.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    auc = test(test_loader, model, args, device)