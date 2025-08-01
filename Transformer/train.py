import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model import Transformer
from data.loader import tokenizer, train_loader, create_src_mask, create_tgt_mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for src, tgt in loop:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src).to(device)
        tgt_mask = create_tgt_mask(tgt_input).to(device)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} finished with loss: {epoch_loss / len(train_loader):.4f}")