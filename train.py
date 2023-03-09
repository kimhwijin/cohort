from torch import nn
import torch
from tqdm import tqdm
from torchmetrics import MetricCollection, F1Score, Accuracy, AUROC, Precision, Recall, ConfusionMatrix
import numpy as np
from collections import defaultdict
import gc
import time
from callback import EarlyStopping

def run_training(config, model, train_dataloader, valid_dataloader, optimizer, device):
    
    start = time.time()

    criterion = nn.CrossEntropyLoss()
    n_epoch = config.epochs
    
    metric_fn = MetricCollection([
        F1Score(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Precision(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Recall(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Accuracy(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        AUROC(task='binary').to(device)
    ])
    early_stopper = EarlyStopping(patience=15)


    best_valid_loss = np.inf
    history = defaultdict(list)

    for epoch in range(n_epoch):
        train_loss = train_step(config, model, train_dataloader, optimizer, criterion, metric_fn, device)
        valid_metric_dict = valid_step(config, model, valid_dataloader, criterion, metric_fn, device)
        
        history['Train Loss'].append(train_loss)
        for key, value in valid_metric_dict.items():
            history[key].append(value)

        # 
        valid_loss = valid_metric_dict['Valid Loss']
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            save_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': valid_metric_dict,
                'epoch': epoch,
                'config': config
            }
            print(f"{config.checkpoint_path} saving.......")
            torch.save(save_state, config.checkpoint_path)
            print(f"{config.checkpoint_path} saved!")

        if early_stopper.early_stop(valid_metric_dict['Valid Loss']):
            print(f"Early Stopping \n epoch: {epoch}/{n_epoch}, Best Validation Loss : {best_valid_loss:.5f}")
            break
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60)
    )

    return history

@torch.no_grad()
def run_predicting(config, model, test_dataloader, device):

    metric_dict = defaultdict(float)
    all_y_prob = np.array([], dtype=np.float32)
    all_y_true = np.array([], dtype=np.float32)
    metric_fn = MetricCollection([
        F1Score(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Precision(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Recall(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        Accuracy(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device),
        AUROC(task='binary').to(device)
    ])

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test ')
    
    for step, data in pbar:
        image, numeric, categorical, label = data
        image = image.to(device);numeric = numeric.to(device);categorical = categorical.to(device);label = label.to(device)

        y_prob = model(image, numeric, categorical).view(-1).sigmoid()
        all_y_prob = np.concatenate([all_y_prob, y_prob.cpu().detach().numpy()])
        all_y_true = np.concatenate([all_y_true, label.cpu().detach().numpy()])

        metric_scores = metric_fn(y_prob, label)
        for key, value in metric_scores.items():
            if 'Binary' in key:
                key = key[6:]
            metric_dict[key] += value.item()
    
    for key, value in metric_dict.items():
        metric_dict[key] = value / (step+1)
    metric_dict['ConfusionMatrix'] = ConfusionMatrix(task='binary', threshold=config.threshold if hasattr(config, 'threshold') else 0.5).to(device)(all_y_prob, all_y_true)
    torch.cuda.empty_cache()
    gc.collect()
    return metric_dict, all_y_prob
    

def train_step(config, model, dataloader, optimizer, criterion, metric_fn, device):
    model.train()
    running_samples = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    metric_dict = defaultdict(float)
    kwargs_metric_dict = defaultdict(float)
    
    for step, data in pbar:
        image, numeric, categorical, label = data
        image = image.to(device);numeric = numeric.to(device);categorical = categorical.to(device);label = label.to(device)
        optimizer.zero_grad()

        logit = model(image, numeric, categorical).view(-1)
        loss = criterion(logit, label)
        loss.backward()

        optimizer.step()

        running_loss += (loss.item() * label.shape[0])
        running_samples += label.shape[0]
        epoch_loss = running_loss / running_samples

        y_prob = logit.sigmoid()
        kwargs_metric_dict['Train Loss'] = np.round(epoch_loss, 4)
        metric_scores = metric_fn(y_prob, label)
        for key, value in metric_scores.items():
            if 'Binary' in key:
                key = key[6:]
            metric_dict[key] += value
            kwargs_metric_dict[key] = np.round(metric_dict[key].item() / (step+1), 3)


        pbar.set_postfix(**kwargs_metric_dict)

        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss

@torch.no_grad()
def valid_step(config, model, dataloader, criterion, metric_fn, device):
    model.eval()
    running_samples = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    metric_dict = defaultdict(float);kwargs_metric_dict = defaultdict(float)

    for step, data in pbar:
        image, numeric, categorical, label = data
        image = image.to(device);numeric = numeric.to(device);categorical = categorical.to(device);label = label.to(device)

        logit = model(image, numeric, categorical).view(-1)
        loss = criterion(logit, label)

        running_loss += (loss.item() * label.shape[0])
        running_samples += label.shape[0]
        epoch_loss = running_loss / running_samples

        y_prob = logit.sigmoid()
        metric_scores = metric_fn(y_prob, label)
        for key, value in metric_scores.items():
            if 'Binary' in key:
                key = key[6:]
            metric_dict[key] += value
            kwargs_metric_dict[key] = np.round(metric_dict[key].item() / (step+1), 3)
        
        kwargs_metric_dict['Valid Loss'] = np.round(epoch_loss, 4)

        pbar.set_postfix(**kwargs_metric_dict)

        torch.cuda.empty_cache()
        gc.collect()

    return kwargs_metric_dict


