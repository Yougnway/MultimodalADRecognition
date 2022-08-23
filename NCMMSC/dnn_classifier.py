import time
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from yacs.config import CfgNode as CN
from sklearn.metrics import (f1_score, precision_score,
                            recall_score, accuracy_score)
from utils.hybrid_dataset import build_hybrid_dataloader
from Nets.DnnClassifier import build_classifier
from sklearn.metrics import confusion_matrix
from transformers import get_cosine_schedule_with_warmup


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_val_epoch(model, optimizer, scheduler, lossfuction, train_loader,
                    val_loader, EPOCHES, DEVICE):
    total_t0 = time.time()
    val_loss_min = 1e10
    # train
    for epoch in range(EPOCHES):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHES))
        print('Training...')
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_loader), elapsed))
            b_inputs = batch[0].to(DEVICE)
            b_labels = batch[1].to(DEVICE)

            optimizer.zero_grad()
            logits = model(b_inputs)
            loss = lossfuction(logits, b_labels)
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)

        # Measure how short this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        # Tracking variables
        total_eval_loss = 0

        y_pred, y_true = [], []
        # Evaluate data for one epoch
        for batch in val_loader:
            b_inputs = batch[0].to(DEVICE)
            b_labels = batch[1].to(DEVICE)
            with torch.no_grad():
                logits = model(b_inputs)
                loss = lossfuction(logits, b_labels)
                # logits = torch.mean(logits, dim=0, keepdim=True)
                # b_labels = b_labels[:1]

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            y_pred += pred_flat.tolist()
            y_true += labels_flat.tolist()
        # Report the final accuracy for this validation run.
        print("  Accuracy: {:.2f}\t"
              "  precision: {:.2f}\t"
              "  f1 score: {:.2f}".format(
                  accuracy_score(y_true, y_pred),
                  precision_score(y_true, y_pred, average='macro'),
                  f1_score(y_true, y_pred, average='macro')))
        print("  confusion matrix: {}".format(
            confusion_matrix(y_true, y_pred).tolist()))

        avg_val_loss = total_eval_loss / len(val_loader)
        if avg_val_loss <= val_loss_min:
            val_loss_min = avg_val_loss
            # save model parameters
        torch.save(model.state_dict(),
                    'save_models/short_task/random3/dnn.pth')
        # Measure how short the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time() - total_t0)))


def test(model, test_loader, lossfuction, DEVICE):
    print("")
    print("Running Test...")
    t0 = time.time()
    state_dict = torch.load('save_models/short_task/random3/dnn.pth')
    model.load_state_dict(state_dict)
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    y_pred, y_true = [], []
    for batch in test_loader:
        b_inputs = batch[0].to(DEVICE)
        b_labels = batch[1].to(DEVICE)
        with torch.no_grad():
            logits = model(b_inputs)
            loss = lossfuction(logits, b_labels)
            # logits = torch.mean(logits, dim=0, keepdim=True)
            # b_labels = b_labels[:1]

        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        y_pred += pred_flat.tolist()
        y_true += labels_flat.tolist()
    # Report the final accuracy for this validation run.
    avg_val_loss = total_eval_loss / len(test_loader)
    print("  Accuracy: {:.4f}\t"
          "  precision: {:.4f}\t"
          "  recall: {:.4f}\t"
          "  f1 score: {:.2f}".format(
              accuracy_score(y_true, y_pred),
              precision_score(y_true, y_pred, average='macro'),
              recall_score(y_true, y_pred, average='macro'),
              f1_score(y_true, y_pred, average='macro')))
    print("  confusion matrix: {}".format(
        confusion_matrix(y_true, y_pred).tolist()))
    # Measure how short the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  test Loss: {0:.2f}".format(avg_val_loss))
    print("  test took: {:}".format(validation_time))


def get_opt():
    cfg = CN()
    cfg.DEVICE = 'cuda:0'
    cfg.DATA = CN()
    cfg.DATA.NUM_WORKERS = 2
    cfg.DATA.BATCH_SIZES = 4
    cfg.DATA.CSV_TRAIN_FILE = 'short_csvs/train.csv'
    cfg.DATA.CSV_VAL_FILE = 'short_csvs/val.csv'
    cfg.DATA.CSV_TEST_FILE = 'short_csvs/test.csv'
    cfg.DATA.FEATURES = ['bert', 'wav2vec', 'is10']
    cfg.DATA.TAILS = ['.text.npy', '.wav.npy', '.IS10_paraling.npy']
    cfg.DATA.FUSION = 'cat'
    cfg.DATA.IS10_NORM = "Data/short_opensmile/IS10_paraling/normalizer.npy"
    cfg.DATA.EGEMAPS_NORM = "Data/short_opensmile/eGeMAPS/normalizer.npy"
    cfg.DATA.COMPARE_NORM = "Data/short_opensmile/ComParE_2016/normalizer.npy"
    cfg.DATA.FEAT_DIRS = ['short_pretrain/random3/script', 'short_pretrain/random3/wav', 'short_opensmile/IS10_paraling']
    cfg.DATA.OLD_DIR = 'short_scripts'
    # classifier
    cfg.CLASSIFIER = CN()
    cfg.CLASSIFIER.IN_DIM = 1966
    cfg.CLASSIFIER.NUM_LAYERS = 2
    cfg.CLASSIFIER.HIDDEN_DIM = 256
    cfg.CLASSIFIER.NUM_CLASSES = 3
    cfg.CLASSIFIER.DROP_RATE = 0.5
    cfg.CLASSIFIER.WEIGHTS = ''
    return cfg


def setup_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    cfg = get_opt()
    DEVICE = torch.device('cuda:0')
    EPOCHES = 32
    LR = 0.0005

    # get dataloader
    dataloaders = build_hybrid_dataloader(cfg)
    train_loader = dataloaders.get('train_loader', None)
    val_loader = dataloaders.get('val_loader', None)
    test_loader = dataloaders.get('test_loader', None)

    # get WavLM classfication model
    lossfuction = nn.CrossEntropyLoss()
    model = build_classifier(cfg)
    # model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # get optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    )
    total_steps = len(train_loader) * EPOCHES
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps)
    # train val
    train_val_epoch(model, optimizer, scheduler, lossfuction, train_loader,
                    val_loader, EPOCHES, DEVICE)
    # test
    test(model, test_loader, lossfuction, DEVICE)
