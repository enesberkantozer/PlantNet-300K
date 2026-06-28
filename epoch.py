import torch
from tqdm import tqdm
from utils import count_correct_topk, count_correct_avgk, update_correct_per_class, \
    update_correct_per_class_topk, update_correct_per_class_avgk

import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.distributed as dist

def gather_tensor(t, world_size):
    gather_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gather_t, t)
    return torch.cat(gather_t)

def train_epoch(model, optimizer, train_loader, criteria, loss_train, acc_train, topk_acc_train, list_k, n_train, use_gpu, scaler, distributed, local_rank, is_main, num_gpus):
    model.train()
    loss_epoch_train = 0
    n_correct_train = 0
    n_correct_topk_train = defaultdict(int)
    topk_acc_epoch_train = {}
    
    all_preds_train = []
    all_targets_train = []

    pbar = tqdm(train_loader, desc='train', position=0, disable=not is_main)
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(pbar):
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(non_blocking=True), batch_y_train.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                batch_output_train = model(batch_x_train)
                loss_batch_train = criteria(batch_output_train, batch_y_train)
            
            loss_epoch_train += loss_batch_train.item()
            scaler.scale(loss_batch_train).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_output_train = model(batch_x_train)
            loss_batch_train = criteria(batch_output_train, batch_y_train)
            loss_epoch_train += loss_batch_train.item()
            loss_batch_train.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(batch_output_train, dim=-1)
            n_correct_train += torch.sum(torch.eq(batch_y_train, preds)).item()
            for k in list_k:
                n_correct_topk_train[k] += count_correct_topk(scores=batch_output_train, labels=batch_y_train, k=k).item()
            
            all_preds_train.extend(preds.cpu().numpy())
            all_targets_train.extend(batch_y_train.cpu().numpy())

    with torch.no_grad():
        if distributed:
            all_preds_t = torch.tensor(all_preds_train).cuda()
            all_targets_t = torch.tensor(all_targets_train).cuda()
            all_preds_t = gather_tensor(all_preds_t, num_gpus)
            all_targets_t = gather_tensor(all_targets_t, num_gpus)
            all_preds_train = all_preds_t.cpu().numpy()
            all_targets_train = all_targets_t.cpu().numpy()

            metrics_tensor = torch.tensor([loss_epoch_train, n_correct_train] + [n_correct_topk_train[k] for k in list_k], dtype=torch.float64).cuda()
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            loss_epoch_train = metrics_tensor[0].item()
            n_correct_train = metrics_tensor[1].item()
            for i, k in enumerate(list_k):
                n_correct_topk_train[k] = metrics_tensor[2 + i].item()

        if is_main:
            loss_epoch_train /= (batch_idx + 1) if not distributed else ((batch_idx + 1) * num_gpus)
            epoch_accuracy_train = n_correct_train / n_train
            for k in list_k:
                topk_acc_epoch_train[k] = n_correct_topk_train[k] / n_train

            loss_train.append(loss_epoch_train)
            acc_train.append(epoch_accuracy_train)
            topk_acc_train.append(topk_acc_epoch_train)
            
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets_train, all_preds_train, average='macro', zero_division=0)
        else:
            epoch_accuracy_train = 0
            precision, recall, f1 = 0, 0, 0

    return loss_epoch_train, epoch_accuracy_train, topk_acc_epoch_train, precision, recall, f1


def val_epoch(model, val_loader, criteria, loss_val, acc_val, topk_acc_val, avgk_acc_val,
              class_acc_val, list_k, dataset_attributes, use_gpu, distributed, local_rank, is_main, num_gpus):
    model.eval()
    with torch.no_grad():
        n_val = dataset_attributes['n_val']
        loss_epoch_val = 0
        n_correct_val = 0
        n_correct_topk_val, n_correct_avgk_val = defaultdict(int), defaultdict(int)
        topk_acc_epoch_val, avgk_acc_epoch_val = {}, {}
        lmbda_val = {}
        class_acc_dict = {}
        class_acc_dict['class_acc'] = defaultdict(int)
        class_acc_dict['class_topk_acc'], class_acc_dict['class_avgk_acc'] = {}, {}
        for k in list_k:
            class_acc_dict['class_topk_acc'][k], class_acc_dict['class_avgk_acc'][k] = defaultdict(int), defaultdict(int)
        
        list_val_proba = []
        list_val_labels = []
        all_preds_val = []

        pbar = tqdm(val_loader, desc='val', position=0, disable=not is_main)
        for batch_idx, (batch_x_val, batch_y_val) in enumerate(pbar):
            if use_gpu:
                batch_x_val, batch_y_val = batch_x_val.cuda(non_blocking=True), batch_y_val.cuda(non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=bool(use_gpu)):
                batch_output_val = model(batch_x_val)
                loss_batch_val = criteria(batch_output_val, batch_y_val)
                
            batch_proba = F.softmax(batch_output_val, dim=-1).float()
            list_val_proba.append(batch_proba)
            list_val_labels.append(batch_y_val)

            loss_epoch_val += loss_batch_val.item()

            preds = torch.argmax(batch_output_val, dim=-1)
            n_correct_val += torch.sum(torch.eq(batch_y_val, preds)).item()
            
            all_preds_val.extend(preds.cpu().numpy())
            for k in list_k:
                n_correct_topk_val[k] += count_correct_topk(scores=batch_output_val, labels=batch_y_val, k=k).item()

        val_probas = torch.cat(list_val_proba)
        val_labels = torch.cat(list_val_labels)

        if distributed:
            val_probas = gather_tensor(val_probas, num_gpus)
            val_labels = gather_tensor(val_labels, num_gpus)
            
            all_preds_t = torch.tensor(all_preds_val).cuda()
            all_preds_t = gather_tensor(all_preds_t, num_gpus)
            all_preds_val = all_preds_t.cpu().numpy()

            metrics_tensor = torch.tensor([loss_epoch_val, n_correct_val] + [n_correct_topk_val[k] for k in list_k], dtype=torch.float64).cuda()
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            loss_epoch_val = metrics_tensor[0].item()
            n_correct_val = metrics_tensor[1].item()
            for i, k in enumerate(list_k):
                n_correct_topk_val[k] = metrics_tensor[2 + i].item()
            
        if is_main:
            class_acc_dict['class_acc'] = defaultdict(int)
            for k in list_k:
                class_acc_dict['class_topk_acc'][k], class_acc_dict['class_avgk_acc'][k] = defaultdict(int), defaultdict(int)
            
            n_correct_avgk_val = defaultdict(int)
            
            update_correct_per_class(val_probas, val_labels, class_acc_dict['class_acc'])
            for k in list_k:
                update_correct_per_class_topk(val_probas, val_labels, class_acc_dict['class_topk_acc'][k], k)

            flat_val_probas = torch.flatten(val_probas)
            sorted_probas, _ = torch.sort(flat_val_probas, descending=True)

            for k in list_k:
                lmbda_val[k] = 0.5 * (sorted_probas[n_val * k - 1] + sorted_probas[n_val * k]).item()
                n_correct_avgk_val[k] += count_correct_avgk(probas=val_probas, labels=val_labels, lmbda=lmbda_val[k]).item()
                update_correct_per_class_avgk(val_probas, val_labels, class_acc_dict['class_avgk_acc'][k], lmbda_val[k])

            loss_epoch_val /= (batch_idx + 1) if not distributed else ((batch_idx + 1) * num_gpus)
            epoch_accuracy_val = n_correct_val / n_val
            for k in list_k:
                topk_acc_epoch_val[k] = n_correct_topk_val[k] / n_val
                avgk_acc_epoch_val[k] = n_correct_avgk_val[k] / n_val
            for class_id in class_acc_dict['class_acc'].keys():
                n_class_val = dataset_attributes['class2num_instances']['val'][class_id]
                class_acc_dict['class_acc'][class_id] /= n_class_val
                for k in list_k:
                    class_acc_dict['class_topk_acc'][k][class_id] /= n_class_val
                    class_acc_dict['class_avgk_acc'][k][class_id] /= n_class_val

            loss_val.append(loss_epoch_val)
            acc_val.append(epoch_accuracy_val)
            topk_acc_val.append(topk_acc_epoch_val)
            avgk_acc_val.append(avgk_acc_epoch_val)
            class_acc_val.append(class_acc_dict)
            
            precision, recall, f1, _ = precision_recall_fscore_support(val_labels.cpu().numpy(), all_preds_val, average='macro', zero_division=0)
        else:
            epoch_accuracy_val, precision, recall, f1 = 0, 0, 0, 0

    return loss_epoch_val, epoch_accuracy_val, topk_acc_epoch_val, avgk_acc_epoch_val, lmbda_val, precision, recall, f1


def test_epoch(model, test_loader, criteria, list_k, lmbda, use_gpu, dataset_attributes, distributed, local_rank, is_main, num_gpus):
    if is_main:
        print()
    model.eval()
    with torch.no_grad():
        n_test = dataset_attributes['n_test']
        loss_epoch_test = 0
        n_correct_test = 0
        topk_acc_epoch_test, avgk_acc_epoch_test = {}, {}
        n_correct_topk_test, n_correct_avgk_test = defaultdict(int), defaultdict(int)

        class_acc_dict = {}
        class_acc_dict['class_acc'] = defaultdict(int)
        class_acc_dict['class_topk_acc'], class_acc_dict['class_avgk_acc'] = {}, {}
        for k in list_k:
            class_acc_dict['class_topk_acc'][k], class_acc_dict['class_avgk_acc'][k] = defaultdict(int), defaultdict(int)

        all_preds_test = []
        all_targets_test = []
        list_test_proba = []

        pbar = tqdm(test_loader, desc='test', position=0, disable=not is_main)
        for batch_idx, (batch_x_test, batch_y_test) in enumerate(pbar):
            if use_gpu:
                batch_x_test, batch_y_test = batch_x_test.cuda(non_blocking=True), batch_y_test.cuda(non_blocking=True)
            with torch.amp.autocast('cuda', enabled=bool(use_gpu)):
                batch_output_test = model(batch_x_test)
                loss_batch_test = criteria(batch_output_test, batch_y_test)
                
            batch_proba_test = F.softmax(batch_output_test, dim=-1).float()
            list_test_proba.append(batch_proba_test)
            loss_epoch_test += loss_batch_test.item()

            preds = torch.argmax(batch_output_test, dim=-1)
            n_correct_test += torch.sum(torch.eq(batch_y_test, preds)).item()
            
            all_preds_test.extend(preds.cpu().numpy())
            all_targets_test.extend(batch_y_test.cpu().numpy())
            for k in list_k:
                n_correct_topk_test[k] += count_correct_topk(scores=batch_output_test, labels=batch_y_test, k=k).item()
        
        test_probas = torch.cat(list_test_proba)
        test_labels = torch.tensor(all_targets_test).cuda()
        
        if distributed:
            test_probas = gather_tensor(test_probas, num_gpus)
            test_labels = gather_tensor(test_labels, num_gpus)
            
            all_preds_t = torch.tensor(all_preds_test).cuda()
            all_preds_t = gather_tensor(all_preds_t, num_gpus)
            all_preds_test = all_preds_t.cpu().numpy()

            metrics_tensor = torch.tensor([loss_epoch_test, n_correct_test] + [n_correct_topk_test[k] for k in list_k], dtype=torch.float64).cuda()
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            loss_epoch_test = metrics_tensor[0].item()
            n_correct_test = metrics_tensor[1].item()
            for i, k in enumerate(list_k):
                n_correct_topk_test[k] = metrics_tensor[2 + i].item()

        if is_main:
            update_correct_per_class(test_probas, test_labels, class_acc_dict['class_acc'])
            for k in list_k:
                update_correct_per_class_topk(test_probas, test_labels, class_acc_dict['class_topk_acc'][k], k)
                
            for k in list_k:
                n_correct_avgk_test[k] += count_correct_avgk(probas=test_probas, labels=test_labels, lmbda=lmbda[k]).item()
                update_correct_per_class_avgk(test_probas, test_labels, class_acc_dict['class_avgk_acc'][k], lmbda[k])

            loss_epoch_test /= (batch_idx + 1) if not distributed else ((batch_idx + 1) * num_gpus)
            epoch_accuracy_test = n_correct_test / n_test
            for k in list_k:
                topk_acc_epoch_test[k] = n_correct_topk_test[k] / n_test
                avgk_acc_epoch_test[k] = n_correct_avgk_test[k] / n_test

            for class_id in class_acc_dict['class_acc'].keys():
                n_class_test = dataset_attributes['class2num_instances']['test'][class_id]
                class_acc_dict['class_acc'][class_id] /= n_class_test
                for k in list_k:
                    class_acc_dict['class_topk_acc'][k][class_id] /= n_class_test
                    class_acc_dict['class_avgk_acc'][k][class_id] /= n_class_test
                    
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels.cpu().numpy(), all_preds_test, average='macro', zero_division=0)
        else:
            epoch_accuracy_test, precision, recall, f1 = 0, 0, 0, 0

    return loss_epoch_test, epoch_accuracy_test, topk_acc_epoch_test, avgk_acc_epoch_test, class_acc_dict, precision, recall, f1
