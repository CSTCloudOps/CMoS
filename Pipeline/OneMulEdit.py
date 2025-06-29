import torch
import toml
import argparse
import torch.nn as nn
import sys
from torchinfo import summary
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PipeAtomic import *

if __name__ == "__main__":
    seed = 2025
    print("seed is %d" %seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    from model.CMoS import Model
    model_name = "CMoS"

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_len', type=int)
    parser.add_argument('--dataset', type=str)
    cmdargs = parser.parse_args()
    dataset = cmdargs.dataset

    # check if specified config exists
    if os.path.exists("./configs/{}_{}.toml".format(dataset, cmdargs.pred_len)):
        config = toml.load("./configs/{}_{}.toml".format(dataset, cmdargs.pred_len))["config"]
    else:
        config = toml.load("./configs/{}.toml".format(dataset))["config"]
    
    config["pred_len"] = cmdargs.pred_len
    args = argparse.Namespace(**config)
    
    model = Model.Model(args)
    
    # summary(model, input_size=(1, args.seq_len, args.c))
    
    if not os.path.exists("./model/{}/cpkt/".format(model_name)):
        os.mkdir("./model/{}/cpkt/".format(model_name))
    
    # define optimizer and step_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    criterion = nn.MSELoss()
    comment = f"{dataset}-{args.pred_len}"
    
    if not os.path.exists("./model/{}/cpkt/{}".format(model_name, comment)):
        os.mkdir("./model/{}/cpkt/{}".format(model_name, comment))
    
    with open(f"./model/{model_name}/cpkt/{comment}/config.toml", "w") as f:
        toml.dump({"config": config}, f)
    
    train_oneset_mts(model, args, optimizer, schedular, criterion, [dataset], model_save_path=f"./model/{model_name}/cpkt/{comment}")
