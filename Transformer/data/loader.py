import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

dataset = load_dataset('wmt14','de-en',split='train[:1%]')
tokenizer = AutoTokenizer.from_pretrained('t5-small')
MAX_LEN = 64
batch_size = 32

def preprocess(example):
    en_text = example['translation']['en']
    de_text = example['translation']['de']

    inputs = tokenizer(en_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    targets = tokenizer(de_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")

    return {
        'src_input_ids': inputs.input_ids.squeeze(0),
        'tgt_input_ids': targets.input_ids.squeeze(0)
    }
def create_src_mask(src):
    mask = (src != 0).unsqueeze(1).unsqueeze(2)
    return mask
def create_tgt_mask(tgt):
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1) 
    return pad_mask & subsequent_mask
def collate_fn(batch):
    src = torch.stack([item['src_input_ids'] for item in batch])
    tgt = torch.stack([item['tgt_input_ids'] for item in batch])
    return src,tgt


processed_dataset = dataset.map(preprocess,remove_columns=dataset.column_names)
processed_dataset.set_format(type="torch", columns=["src_input_ids", "tgt_input_ids"])
train_loader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)