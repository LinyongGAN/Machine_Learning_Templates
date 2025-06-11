import torch
import torch.nn as nn
import math

#位置编码与Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class PoTSTransformer(nn.Module):
    # Constructor
    def __init__(self, d_input, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout_p, layer_norm_eps, padding_idx = None, pos_expansion_dim = 64):
        
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx

        # EDITED - nn.Embedding must be replaced with a linear layer - separate for src and tgt as
        # they are not anymore "one-hot-encoded" tokens
        #
        #if padding_idx != None:
        #    # Token embedding layer - this takes care of converting integer to vectors
        #    self.embedding = nn.Embedding(num_tokens+1, d_model, padding_idx = self.padding_idx)
        #else:
        #    # Token embedding layer - this takes care of converting integer to vectors
        #    self.embedding = nn.Embedding(num_tokens, d_model)
        #
        self.embedding = nn.Linear(d_input, d_model)
        
        # EDITED - Token "unembedding" to one-hot token vector 
        #
        #self.unembedding = nn.Linear(d_model, num_tokens)
        self.unembedding = nn.Linear(d_model, d_input)

        # Positional encoding expansion
        self.pos_expansion = nn.Linear(d_model, pos_expansion_dim)
        self.pos_unexpansion = nn.Linear(pos_expansion_dim, d_model)
        

        # Positional encoding
        self.positional_encoder = PositionalEncoding(d_model=pos_expansion_dim, dropout=dropout_p)

        # nn.Transformer that does the magic
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout_p,
            layer_norm_eps = layer_norm_eps,
            norm_first = True
        )

    def forward(self, src, tgt, tgt_mask = None, src_key_padding_mask = None, tgt_key_padding_mask = None):
        # Note: src & tgt default size is (seq_length, batch_num, feat_dim)

        # Token embedding EDITED
        #
        #src = self.src_embedding(src) * math.sqrt(self.d_model)
        #tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        
        # Positional encoding - this is sensitive that data _must_ be seq len x batch num x feat dim
        # Inference often misses the batch num
        if src.dim() == 2: # seq len x feat dim
            src = torch.unsqueeze(src,1) 
        src = self.pos_expansion(src)
        src = self.positional_encoder(src)
        src = self.pos_unexpansion(src)

        if tgt.dim() == 2: # seq len x feat dim
            tgt = torch.unsqueeze(tgt,1) 
        tgt = self.pos_expansion(tgt)
        tgt = self.positional_encoder(tgt)
        tgt = self.pos_unexpansion(tgt)

        # Transformer output
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask = src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        out = self.unembedding(out)
        
        return out