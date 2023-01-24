import random
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.datasets import fetch_20newsgroups
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch.nn as nn
torch.cuda.empty_cache()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)
set_seed(1)


model_name = "EleutherAI/gpt-j-6B"
max_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
test_size=0.2


dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
documents = dataset.data
labels = dataset.target
(train_texts, valid_texts, train_labels, valid_labels), target_names = train_test_split(documents, labels, test_size=test_size), dataset.target_names
x_train, y_train, x_val, y_val = train_texts, valid_texts, train_labels, valid_labels

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(y_train, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
          self.encodings = encodings
          self.labels = labels

      def __getitem__(self, idx):
          global item
          item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
          item["labels"] = torch.tensor([self.labels[idx]])
          return item

      def __len__(self):
          global len
          return len(self.labels)

train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
# print(train_encodings)
model = AutoModelForCausalLM.from_pretrained(model_name, num_labels=2)
# model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
# model.to(f'cuda:{model.device_ids[0]}
model = nn.DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())]) #auto pic number of gpu
# model = model.cuda(model.device_ids=[0])


from sklearn.metrics import accuracy_score


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  #     calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  #recall = recall_score(labels, preds)
  #precision = precision_score(labels, preds)
  #f1 = f1_score(labels, preds)
  return {
      'accuracy': acc,
  }



def train_model(train_dataset, valid_dataset):

  ##################################Train ARGUMENTS###############################
  training_args = TrainingArguments(
  output_dir='./results',

  ################change number of epochs for testing to 3 , for actual training 1000
  num_train_epochs=100,
  per_device_train_batch_size=128,
  per_device_eval_batch_size=64,
  warmup_steps=500,
  weight_decay=0.01,
  learning_rate=0.00005,
  logging_dir='./logs',
  load_best_model_at_end=True,

  logging_steps=400,
  save_steps=400,
  evaluation_strategy="steps",
  metric_for_best_model = 'f1'
  )
  #################################################################################

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=valid_dataset,
      compute_metrics=compute_metrics,
      # callbacks = [EarlyStopping(monitor='val_loss',mode='min')]
      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

  )

  return trainer


trainer=train_model(train_dataset, valid_dataset)
trainer.train()

train_model(train_dataset, valid_dataset).evaluate()

model_path = "bert_train"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
final_x = torch.save(model.state_dict(), 'bert-model.bin') #can remove if throws error
