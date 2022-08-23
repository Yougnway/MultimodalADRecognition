import time
import torch
import datetime
import numpy as np
import random
import torch.nn as nn
from Nets.WavClassifier import WavClassifier
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from utils.audio_dataset import get_audioloader


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


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
            b_input_ids = batch[0].to(DEVICE)
            b_labels = batch[1].to(DEVICE)

            optimizer.zero_grad()
            logits = model(b_input_ids)
            loss = lossfuction(logits, b_labels)
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)

        # Measure how long this epoch took.
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
        total_eval_accuracy = 0
        total_eval_loss = 0

        y_pred, y_true = [], []
        # Evaluate data for one epoch
        for batch in val_loader:
            b_input_ids = batch[0].to(DEVICE)
            b_labels = batch[1].to(DEVICE)
            with torch.no_grad():
                logits = model(b_input_ids)
                loss = lossfuction(logits, b_labels)
                # no multi-clip
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
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        print("  Accuracy: {:.2f}\t"
              "  precision: {:.2f}\t"
              "  f1 score: {:.2f}".format(
                  avg_val_accuracy,
                  precision_score(y_true, y_pred, average='macro'),
                  f1_score(y_true, y_pred, average='macro')))
        print("  confusion matrix: {}".format(
            confusion_matrix(y_true, y_pred).tolist()))

        avg_val_loss = total_eval_loss / len(val_loader)
        if avg_val_loss <= val_loss_min:
            val_loss_min = avg_val_loss
            # save model parameters
            # torch.save(model.state_dict(),
            #            'save_models/audio.pth')
            torch.save(model.state_dict(),
                       'save_models/short_task/random3/wav2vec.pth')
        # Measure how long the validation run took.
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
    state_dict = torch.load('save_models/short_task/random3/wav2vec.pth')
    model.load_state_dict(state_dict)
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    y_pred, y_true = [], []
    for batch in test_loader:
        b_input_ids = batch[0].to(DEVICE)
        b_labels = batch[1].to(DEVICE)
        with torch.no_grad():
            # print(b_input_ids.shape)
            logits = model(b_input_ids)
            loss = lossfuction(logits, b_labels)
            # no multi-clip
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
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    avg_val_loss = total_eval_loss / len(test_loader)
    print("  Accuracy: {:.4f}\t"
          "  precision: {:.4f}\t"
          "  recall: {:.4f}"
          "  f1 score: {:.2f}".format(
              accuracy_score(y_true, y_pred),
              precision_score(y_true, y_pred, average='macro'),
              recall_score(y_true, y_pred, average='macro'),
              f1_score(y_true, y_pred, average='macro')))
    print("  confusion matrix: {}".format(
        confusion_matrix(y_true, y_pred).tolist()))
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  test Loss: {0:.2f}".format(avg_val_loss))
    print("  test took: {:}".format(validation_time))


seed_val = 32

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0')
EPOCHES = 20
LR = 1e-5

# get dataloader
train_loader = get_audioloader('short_csvs/train.csv')
val_loader = get_audioloader('short_csvs/val.csv',
                             shuffle=False, train=False)
test_loader = get_audioloader('short_csvs/test.csv',
                             shuffle=False, train=False)

# get WavLM classfication model
lossfuction = nn.CrossEntropyLoss()
model = WavClassifier(3)
# model = nn.DataParallel(model)
model = model.to(DEVICE)

# get optimizer
optimizer = AdamW(
    model.parameters(),
    lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    eps=1e-8  # args.adam_epsilon  - default is 1e-8.
)
total_steps = len(train_loader) * EPOCHES
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Default value in run_glue.py
    num_training_steps=total_steps)

# train val
train_val_epoch(model, optimizer, scheduler, lossfuction, train_loader,
                val_loader, EPOCHES, DEVICE)
# test
test(model, test_loader, lossfuction, DEVICE)
