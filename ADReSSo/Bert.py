from transformers import get_linear_schedule_with_warmup
from utils.cls_dataset import get_dataloader
from Nets.BertClassifier import BertClassifier
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
import time
import torch
import datetime
import numpy as np
import random


seed_val = 49

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


DEVICE = torch.device('cuda:0')
EPOCHES = 3
# LR = 3e-5
LR = 3e-5
FOLD_N = 9

# get dataloader
train_loader = get_dataloader('nfoldsplits/train_'+str(FOLD_N)+'.csv')
val_loader = get_dataloader('nfoldsplits/val_'+str(FOLD_N)+'.csv', shuffle=False)

# get bert classfication model
model = BertClassifier()

model = nn.DataParallel(model)
lossfuction = nn.CrossEntropyLoss()

model = model.to(DEVICE)

# get optimizer
optimizer = AdamW(model.parameters(),
                  lr = LR,   # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
total_steps = len(train_loader) * EPOCHES
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
training_stats = []
total_t0 = time.time()
# train
for epoch in range(EPOCHES):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHES))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_loader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

        # `batch` contains three pytorch tensors:
        #   [0]: txt
        #   [1]: input ids 
        #   [2]: attention masks
        #   [3]: labels 
        b_input_ids = batch[1].to(DEVICE)
        b_input_mask = batch[2].to(DEVICE)
        b_labels = batch[3].to(DEVICE)

        optimizer.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        logits = model(b_input_ids, 
                       attention_mask=b_input_mask)
        loss = lossfuction(logits, b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
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
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_loss = 0
    nb_eval_steps = 0

    y_pred, y_true = [], []

    # Evaluate data for one epoch
    for batch in val_loader:
        
        # `batch` contains three pytorch tensors:
        #   [0]: txt
        #   [1]: input ids 
        #   [2]: attention masks
        #   [3]: labels 
        b_input_ids = batch[1].to(DEVICE)
        b_input_mask = batch[2].to(DEVICE)
        b_labels = batch[3].to(DEVICE)
        
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = model(b_input_ids, 
                           attention_mask=b_input_mask)
            loss = lossfuction(logits, b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        y_pred += pred_flat.tolist()
        y_true += labels_flat.tolist()

    print("  Accuracy: {:.2f}\t"
          "  precision: {:.2f}\t"
          "  f1 score: {:.2f}\t"
          "  confusion matrix: {}".format(accuracy_score(y_true, y_pred),
                                      precision_score(y_true, y_pred),
                                      f1_score(y_true, y_pred),
                                      confusion_matrix(y_true, y_pred).tolist()))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_loader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch + 1,
            # 'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': accuracy_score(y_true, y_pred),
            # 'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
# print("Saving model to: ", 'save_models/'+SPLIT+'/bert/bert-'+str(FOLD_N)+'.pth')
# torch.save(model.module.state_dict(), 'save_models/'+SPLIT+'/bert/bert-'+str(FOLD_N)+'.pth')
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
