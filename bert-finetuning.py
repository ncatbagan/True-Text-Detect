import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json

# load and prepare dataset
with open('data1.json', 'r') as file:
    data = json.load(file)
texts = [item['text'] for item in data]
labels = torch.tensor([item['label'] for item in data])

# load and tokenise texts & labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# create DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=16)

# initialise BERT
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# fine-tune BERT on dataset
optimizer = AdamW(bert_model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(3):  # train for 3 cycles (epochs)
    for batch in loader:
        input_ids, attention_mask, batch_labels = batch
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# save fine-tuned model
bert_model.save_pretrained('./fine_tuned_bert/')