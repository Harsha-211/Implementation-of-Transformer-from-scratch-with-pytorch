import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32, device=Q.device)
        )
        if mask is not None:
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model //num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. Linear projections: (batch, seq_len, d_model) → (batch, seq_len, num_heads * d_k)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
         # 2. Split into heads: (batch, seq_len, num_heads, d_k)

        Q = Q.view(batch_size , -1, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, -1, self.num_heads , self.d_k).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads , self.d_k).transpose(1,2)

        # 3. Apply attention on each head
        # Now Q, K, V: (batch, heads, seq_len, d_k)
        
        atten_output, atten_weights = self.attention(Q,K,V,mask = mask)

        # 4. Concatenate heads
        # (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        
        concat =  atten_output.transpose(1,2).contiguous().view(batch_size , -1, self.d_model)

        # 5. Final linear projection
        
        output = self.W_o(concat)

        return output , atten_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model , d_ff , dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model , d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x,x,x,mask)
        x = self.norm1(self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask = None):
        _x, _ = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x+self.dropout1(_x))

        _x, _ = self.cross_attn(x,enc_out, enc_out, memory_mask)
        x = self.norm2(x + self.dropout2(_x))

        ffn_out = self.ffn(x)
        x = self.norm3(x+self.dropout3(ffn_out))

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, 
                 num_encoder_layers, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, memory_mask)
        logits = self.final_linear(out)
        return logits