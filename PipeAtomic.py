import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import os
import numpy as np
from TSDataset import *
from torch.utils.data import SubsetRandomSampler
import tqdm
import toml
import time
import itertools

from utils import EarlyStoppingBySlope

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('my_logger')

# def fix_seed(seed=None):
#     if not seed:
#         seed = random.randint(1, 10000)
#     print("seed is %d" %seed)

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

def set_logger(model_save_path):
    folder_path = model_save_path
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_handler = logging.FileHandler('{}/log.txt'.format(folder_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(ch)

cuda = True
if cuda == True and torch.cuda.is_available():
    device = torch.device("cuda")
    print("=== Using CUDA ===")
else:
    if cuda == True and not torch.cuda.is_available():
        print("=== CUDA is unavailable ===")
    device = torch.device("cpu")
    print("=== Using CPU ===")


def train_oneset_mts(model, config, optimizer, schedular, criterion, dataset, model_save_path, dtype=torch.float32):
    """
    config(args):
        seq_len: int
        pred_len: int
        epochs: int
        batch_size: int
    """
    # if model_save_path:
        # set_logger(model_save_path)
    # logger.info(f">>> One set {dataset} train phase")

    
    with open(f"{model_save_path}/log.txt", "a") as f:
        f.write(f"========\n{vars(config)}\n")
    
    model = model.to(device).to(dtype)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if dataset[0][:3] == "ETT":
        train_prop = 0.6
        valid_prop = 0.2
    
    else:
        train_prop = 0.7
        valid_prop = 0.1
    
    train_loader = torch.utils.data.DataLoader(
        MTSDataset(dataset, config.seq_len, config.pred_len, phase="train", train_prop=train_prop, valid_prop=valid_prop), 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        MTSDataset(dataset, config.seq_len, config.pred_len, phase="valid", train_prop=train_prop, valid_prop=valid_prop), 
        batch_size=config.batch_size * 2, 
        shuffle=False,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        MTSDataset(dataset, config.seq_len, config.pred_len, phase="test", train_prop=train_prop, valid_prop=valid_prop), 
        batch_size=config.batch_size * 2, 
        shuffle=False,
        drop_last=True
    )
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    early_stopper = EarlyStoppingBySlope(window_size=5, slope_threshold=-0.001, patience=5)

    patience = 30
    min_valid_loss = float('inf')
    epochs_no_improve = 0
    valid_losses = []
    test_losses = []
    maes = []
    
    for epoch in range(config.epochs):
        model.train(mode=True)
        train_loss, valid_loss, avg_loss = 0,0,0
        
        # Training
        loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
        # loop = enumerate(train_loader)
        for idx, (seq, pred) in loop:
            optimizer.zero_grad()
            
            seq = seq.to(device).to(dtype)
            pred = pred.to(device).to(dtype)
            
            output = model(seq)
            loss = criterion(output, pred)
                
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.cpu().item()
            # loop.set_description(f'Training Epoch [{epoch}/{config.epochs}]')
            # loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            train_loss = avg_loss/(idx+1)

        # Validation
        model.eval()
        avg_loss = 0
        loop = enumerate(valid_loader)
        with torch.no_grad():
            for idx, (seq, pred) in loop:
                seq = seq.to(device).to(dtype)
                pred = pred.to(device).to(dtype)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(seq)
                    loss = nn.MSELoss()(output, pred)
                    
                avg_loss += loss.cpu().item()
                # loop.set_description(f'Validation Epoch [{epoch}/{config.epochs}]')
                # loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                valid_loss = avg_loss/(idx+1)
        
        # Test 
        avg_loss = 0
        test_loss = 0
        # loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        loop = enumerate(test_loader)
        gts = []
        prs = []
        with torch.no_grad():
            for idx, (seq, pred) in loop:
                seq = seq.to(device).to(dtype)
                pred = pred.to(device).to(dtype)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(seq)
                    loss = nn.MSELoss()(output, pred)
                    prs.append(output.cpu().numpy())
                    gts.append(pred.cpu().numpy())
                    
                avg_loss += loss.cpu().item()
                test_loss = avg_loss/(idx+1)
            
        schedular.step()
         
        prs = np.concatenate(prs, axis=0)
        gts = np.concatenate(gts, axis=0)
        test_loss = np.mean((prs - gts) ** 2)
        mae = np.mean(np.abs(prs - gts))
        with open(f"{model_save_path}/log.txt", "a") as f:
            f.write(f"Epoch {epoch} test loss: {test_loss:.6f}, mae: {mae:.6f}, train_loss: {train_loss:.6f} valid loss: {valid_loss:.6f}\n")
        # print(f"Epoch {epoch} test loss: {test_loss}, mae: {mae}, train_loss: {train_loss} valid loss: {valid_loss}")
        
        del prs, gts
        
        test_losses.append(test_loss)
        maes.append(mae)

        criterion_loss = valid_loss
        if (criterion_loss) < min_valid_loss:
            min_valid_loss = (criterion_loss)
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_save_path}/best.pth")
            print(f"Epoch {epoch} : Validation loss decreased, update best model...")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.\n")
                break

        valid_losses.append(criterion_loss)
    
    # find the min test loss and index
    valid_losses = np.array(valid_losses)
    min_idx = np.argmin(valid_losses)
    with open(f"{model_save_path}/log.txt", "a") as f:
        f.write(f"\n\n===Best Results===\nBest Result: Epoch: {min_idx}, Test loss: {test_losses[min_idx]}, MAE:{maes[min_idx]}\n")
    return 
