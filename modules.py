import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset

# Input Embedding
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        return self.embedding(x) * scale

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0, dtype=torch.float32)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

# Normalization Layer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        norm = self.alpha * (x - mean) / std + self.bias
        return norm

# Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by the number of heads"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.heads * self.d_k)
        output = self.out(output)
        return output

# Residual Connection
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, heads, dropout)
        self.multi_head_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        self.residual_connection = ResidualConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        x = self.residual_connection(x, lambda x: self.masked_multi_head_attention(x, x, x, mask))
        x = self.residual_connection(x, lambda x: self.multi_head_attention(x, x, x))
        x = self.residual_connection(x, self.feed_forward)
        return x

# Decoder
class DecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        # Input embedding layer
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # Apply input embeddings
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        x = self.output_layer(x)
        
        return x
    
    def generate_text(self, x):
        x = self.forward(x)
        return x.argmax(dim=-1)
    
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.output_layer(x)
    
    def generate_text(self, x):
        x = self.forward(x)
        return x.argmax(dim=-1)
    
class PositionalModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return self.output_layer(x)
    
    def generate_text(self, x):
        x = self.forward(x)
        return x.argmax(dim=-1)

# Projection Layer
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.linear(x).log_softmax(dim=-1)

# Custom dataset for text data
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, window_size=100, step_size=25):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        # Iterate over texts and create input-target pairs
        for text in texts:
            for i in range(0, len(text) - window_size, step_size):
                input_txt = text[i:i + window_size]
                target_txt = text[i + 1:i + window_size + 1]
                self.input_ids.append(tokenizer.encode(input_txt))
                self.target_ids.append(tokenizer.encode(target_txt))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]), torch.LongTensor(self.target_ids[idx])



    # Tokenizer
class Tokenizer:
    def __init__(self, texts):
        self.vocab = self.create_vocab(texts)
        self.max_len = max([len(text.split()) for text in texts])

    def create_vocab(self, texts):
        vocab = torchtext.vocab.build_vocab_from_iterator([text.split() for text in texts], specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def tokenize(self, texts, vocab, max_len):
        tokenized_texts = []
        for text in texts:
            tokens = text.split()
            token_ids = [vocab['<bos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
            if len(token_ids) < max_len:
                token_ids = token_ids + [vocab['<pad>']] * (max_len - len(token_ids))
            else:
                token_ids = token_ids[:max_len]
            tokenized_texts.append(token_ids)
        return tokenized_texts

    def encode(self, text):
        tokens = text.split()
        token_ids = [self.vocab[token] for token in tokens]
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return token_ids

    def decode(self, token_ids):
        tokens = [self.vocab.lookup_token(token_id) for token_id in token_ids]
        return ' '.join(tokens)
    
    def bos_id(self):
        return self.vocab['<bos>']
    
    def eos_id(self):
        return self.vocab['<eos>']
    
    def pad_id(self):
        return self.vocab['<pad>']
    
    def unk_id(self):
        return self.vocab['<unk>']
    
    def vocab_size(self):
        return len(self.vocab)
    
    def max_length(self):
        return self.max_len