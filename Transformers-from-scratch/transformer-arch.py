# Write your code here :-)
import torch
import torch.nn
import math

class InputEmbeddings(nn.Module):
    #d_model= 512
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size
        self.embedding= nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_len
        self.dropout= nn.Dropout(dropout)

        #create position embeddnig matrix
        pe= torch.zeros(self.seq_len, self.d_model)
        #create a vector (tensor of shape(seq_len, 1)
        position= torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_model, 2), float()* (-math.log(10000.0)/d_model))

        pe[:,0::2]= torch.sin(position*div_term)
        pe[:,1::2]= torch.cos(position*div_term)

        # change shape to (1,seq_len, d_model) # create batch
        pe= pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float= 10**-6):
        self.epsilon = epsilon
        self.alpha= nn.Parameter(torch.ones(1)) # multiply
        self.bias= nn.Parameter(torch.zeros(1)) # add

    def forward(self, x):
        #usually the mean cancels the dimension to which it is applied.. so we want to keep it
        mean= x.mean(dim= -1, keepdim= True)
        std= x.std(dim= -1, keepdim= True)

        return self.alpha*(x-mean)/(std +self.epsilon) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.linear_1= nn.Linear(d_model, dff)
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(dff, d_model)

    def forward(self, x):
        #x= (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model= d_model
        self.h= h

        assert d_model%h ==0, "d_model is not divisible by h"
        self.dk= d_model//h
        self.wq= nn.Linear(d_model, d_model)
        self.wk= nn.Linear(d_model, d_model)
        self.wv= nn.Linear(d_model, d_model)

        self.wo= nn.Linear(d_model, d_model)
        self.dropout= nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, nn.Dropout):
        dk= query.shape[-1]
        attention_scores= (query@key.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores= attention_scores.softmax(dim= -1)
        if dropout is not None:
            attention_scores= dropout(attention_scores)
        return (attention_scores@value)

    def forward(self, q, k, v, mask):
        q_prime= self.wq(q)
        k_prime= self.wk(k)
        v_prime= self.wv(v)

        #split into heads
        #(batch, seq_len, d_model) -> (batch, seq_len, h, dk)
        q_prime= q_prime.view(q_prime.shape[0], q_prime.shape[1], self.h, self.dk)
        k_prime= k_prime.view(k_prime.shape[0], k_prime.shape[1], self.h, self.dk)
        v_prime= v_prime.view(v_prime.shape[0], v_prime.shape[1], self.h, self.dk)

        #(batch, seq_len, h, dk) -> (batch, h, seq_len, dk)
        q_prime.transpose(1,2)
        k_prime.transpose(1,2)
        v_prime.transpose(1,2)

        x= MultiHeadAttentionBlock.attention(q_prime, k_prime, v_prime, mask, self.dropout)
        #(batch, h, seq_len, dk) -> (batch, seq_len, h, dk)
        x= x.transpose(1,2)
        #(batch, seq_len, h, dk) -> (batch, seq_len, d_model)
        x=x.contiguous().view(x.shape[0], -1, self.h* self.dk) # d_model= h*dk

        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.wo(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm= LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block= self_attention_block
        self.feed_forward_block= feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connections[1](self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(X)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block= self_attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block= feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x= self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,trg_mask))
        x= self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x= self.residual_connections[2](self.feed_forward_block)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()

    def forward(self, x, encoder_ouptut, src_mask, trg_mask):
        for layer in self.layers:
            x= layer(x, encoder_ouptut, src_mask, trg_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj= nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim= -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, trg_embed: InputEmbeddings, src_pos: PositionalEmbedding, trg_pos: PositionalEmbedding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder= encoder
        self.decoder= decoder
        self.src_embed= src_embed
        self.trg_embed= trg_embed
        self.src_pos= src_pos
        self.trg_pos= trg_pos
        self.projection_layer= projection_layer

    def encode(self, src, src_mask):
        src= self.src_embed(src)
        src= self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_ouptut, src_mask, trg, trg_mask):
        trg= self.trg_embed)(trg)
        trg= self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, dff: int = 2048):
    src_embed= InputEmbeddings(d_model, src_vocab_size)
    trg_embed= InputEmbeddings(d_model, trg_vocab_size)

    src_pos= PositionalEmbedding(d_model, src_seq_len, dropout)
    trg_pos= PositionalEmbedding(d_model, trg_seq_len, dropout)

    encoder_blocks= []
    for _ in range(N):
        encoder_self_attention_block= MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block= FeedForwardBlock(d_model, dff, dropout)
        encoder_block= EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks= []
    for _ in range(N):
        decoder_self_attention_block= MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block= MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block= FeedForwardBlock(d_model, dff, dropout)
        decoder_block= DecoderBlock(decoder_cross_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder= nn.ModuleList(encoder_blocks)
    decoder= nn.ModuleList(decoder_blocks)

    projection_layer= ProjectionLayer(d_model, trg_vocab_size)

    transformer= Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
