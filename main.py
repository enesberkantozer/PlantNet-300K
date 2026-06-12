import os
from tqdm import tqdm
import pickle
import argparse
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from epoch import train_epoch, val_epoch, test_epoch
from cli import add_all_parsers
import pandas as pd
import matplotlib.pyplot as plt

def train(args):
    set_seed(args, use_gpu=torch.cuda.is_available())
    train_loader, val_loader, test_loader, dataset_attributes = get_data(args.root, args.image_size, args.crop_size,
                                                                         args.batch_size, args.num_workers, args.pretrained)

    model = get_model(args, n_classes=dataset_attributes['n_classes'])
    criteria = CrossEntropyLoss()

    if args.use_gpu:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.mu, nesterov=True)

    # Containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val = [], [], [], [], []
    
    prec_train, rec_train, f1_train = [], [], []
    prec_val, rec_val, f1_val = [], [], []

    save_name = args.save_name_xp.strip()
    save_dir = os.path.join(os.getcwd(), 'results', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('args.k : ', args.k)

    lmbda_best_acc = None
    best_val_acc = float('-inf')

    for epoch in tqdm(range(args.n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=args.epoch_decay, epoch=epoch)

        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train, p_train, r_train, f_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, args.k,
                                                                              dataset_attributes['n_train'],
                                                                              args.use_gpu)

        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, lmbda_val, p_val, r_val, f_val = val_epoch(model, val_loader, criteria,
                                                  loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                  class_acc_val, args.k, dataset_attributes, args.use_gpu)

        prec_train.append(p_train)
        rec_train.append(r_train)
        f1_train.append(f_train)

        prec_val.append(p_val)
        rec_val.append(r_val)
        f1_val.append(f_val)

        # save model at every epoch
        save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights.tar'))

        # save model with best val accuracy
        if acc_epoch_val > best_val_acc:
            best_val_acc = acc_epoch_val
            lmbda_best_acc = lmbda_val
            save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

        # Create DataFrame and save CSV
        df = pd.DataFrame({
            'epoch': list(range(epoch + 1)),
            'train_loss': loss_train,
            'train_acc': acc_train,
            'train_precision': prec_train,
            'train_recall': rec_train,
            'train_f1': f1_train,
            'val_loss': loss_val,
            'val_acc': acc_val,
            'val_precision': prec_val,
            'val_recall': rec_val,
            'val_f1': f1_val
        })
        df.to_csv(os.path.join(save_dir, save_name + '_metrics.csv'), index=False)
        
        # Plotting
        plt.figure(figsize=(15, 10))
        metrics_to_plot = [('Loss', loss_train, loss_val), ('Accuracy', acc_train, acc_val),
                           ('Precision', prec_train, prec_val), ('Recall', rec_train, rec_val),
                           ('F1 Score', f1_train, f1_val)]
        for i, (name, t_metric, v_metric) in enumerate(metrics_to_plot):
            plt.subplot(2, 3, i+1)
            plt.plot(t_metric, label=f'Train {name}')
            plt.plot(v_metric, label=f'Val {name}')
            plt.title(name)
            plt.xlabel('Epoch')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name + '_metrics_plot.png'))
        plt.close()

        print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'Train -> loss: {loss_epoch_train:.4f} | acc: {acc_epoch_train:.4f} | p: {p_train:.4f} | r: {r_train:.4f} | f1: {f_train:.4f}')
        print(f'Val   -> loss: {loss_epoch_val:.4f} | acc: {acc_epoch_val:.4f} | p: {p_val:.4f} | r: {r_val:.4f} | f1: {f_val:.4f}')

    # load weights corresponding to best val accuracy and evaluate on test
    load_model(model, os.path.join(save_dir, save_name + '_weights_best_acc.tar'), args.use_gpu)
    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test, p_test, r_test, f_test = test_epoch(model, test_loader, criteria, args.k,
                                                  lmbda_best_acc, args.use_gpu,
                                                  dataset_attributes)

    # Save test metrics
    test_df = pd.DataFrame({
        'test_loss': [loss_test_ba],
        'test_acc': [acc_test_ba],
        'test_precision': [p_test],
        'test_recall': [r_test],
        'test_f1': [f_test]
    })
    test_df.to_csv(os.path.join(save_dir, save_name + '_test_metrics.csv'), index=False)
    
    # Plot test metrics as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [acc_test_ba, p_test, r_test, f_test], color=['blue', 'green', 'orange', 'red'])
    plt.title('Test Metrics')
    plt.ylim([0, 1])
    plt.savefig(os.path.join(save_dir, save_name + '_test_metrics_plot.png'))
    plt.close()

    # Save the results as a dictionary and save it as a pickle file in desired location

    results = {'loss_train': loss_train, 'acc_train': acc_train, 'topk_acc_train': topk_acc_train,
               'prec_train': prec_train, 'rec_train': rec_train, 'f1_train': f1_train,
               'loss_val': loss_val, 'acc_val': acc_val, 'topk_acc_val': topk_acc_val, 'class_acc_val': class_acc_val,
               'avgk_acc_val': avgk_acc_val, 'prec_val': prec_val, 'rec_val': rec_val, 'f1_val': f1_val,
               'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test,
                                'precision': p_test,
                                'recall': r_test,
                                'f1_score': f_test},
               'params': args.__dict__}

    with open(os.path.join(save_dir, save_name + '.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)
