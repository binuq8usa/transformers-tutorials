import torch
import torch.nn as nn
import math
from torch.nn.functional import log_softmax, pad, softmax

# Three basic modules

# MultiHeaded Attention - Self Attention and Cross Attention
# FeedForwrd Module - MLP 
# Embeddings module- learnable or lut
# Position Embedding module

# Define the transformer overall schema - encoder-decoder architecture
class Transformer(nn.Module):
    def __init__(self, encoder : nn.Module, decoder : nn.Module, src_embed : nn.Module, tgt_embed : nn.Module, generator: nn.Module):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def encode(self, src, use_mask):
        return self.encoder(self.src_embed(src), use_mask)
    
    # memory is the encoder output
    def decode(self, memory, use_src_mask, tgt, use_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, use_src_mask, use_tgt_mask)
    
    def forward(self, src, tgt, use_src_mask, use_tgt_mask):
        return self.decode(memory=self.encode(src, use_src_mask),
                           tgt=tgt, src_mask=use_src_mask, tgt_mask=use_tgt_mask)


# Define the various blocks
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size) # projecting d_model vector into a vocabulary
    
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1) # along the vocab dimension
        
# define the LayerNorm and cloning to create multiple encoder/decoder blocks
class LayerNorm(nn.Module):
    def __init__(self, feature_size : int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.feature_size = feature_size
        self.gamma = nn.Parameter(torch.ones(feature_size))
        self.beta = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps
    
    def forward(self, x):
        
        assert x.shape[-1] == self.feature_size
        
        x_mean = x.mean(dim=-1, keepdim = True) #  x feature_size
        x_std = x.std(dim=-1, keepdim=True) # 1 x feature_size
        
        # x_mean, x_std is of shape (batch_size x sequence_length x 1)
        # self.gamma, self.beta is of (self.feature_size,) where it scales/shifts each feature dimension
        return self.gamma * (x - x_mean) / (x_std + self.eps) + self.beta

# define the encoder block
import copy
def clones(module: nn.Module, N : int):
    return nn.ModuleList([ copy.deepcopy(module) for _ in range(N) ])
    
class SubLayerConnection(nn.Module):
    def __init__(self, d_model : int, p_dropout : float):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x, sublayer : nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, p_dropout=0.1, d_k = None, d_v = None):
        super(MultiHeadedAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_k is None or d_v is None:
            assert d_model % num_heads == 0 
            self.d_k = d_model // num_heads
            self.d_v = d_model // num_heads
        else:
            self.d_k = d_k
            self.d_v = d_v
        
        # query, key and Value weight matrices
        # self.att_weights = clones(nn.Linear(self.d_model, self.d_model), 3)
        self.att_weights = nn.ModuleDict({
            'W_Q' : nn.Linear(self.d_model, self.num_heads * self.d_k),
            'W_K' : nn.Linear(self.d_model, self.num_heads * self.d_k),
            'W_V' : nn.Linear(self.d_model, self.num_heads * self.d_v)
        })
        self.proj_weights = nn.Linear(self.num_heads * self.d_v, self.d_model)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, input_query_x, input_key_x, input_value_x, use_mask=False):
        # get batch size, sequence length and d_model
        N, L, d_model = input_query_x.shape # maybe shape
        assert d_model == self.d_model
        assert input_key_x.shape[-1] == self.d_model
        
        # d_model = d_k = d_v (d_k shape of query and key, and d_v dimension of value)
        assert input_value_x.shape[-1] == self.d_model

        # get the query, key and value matrices
        query_x = self.att_weights.W_Q(input_query_x)
        key_x = self.att_weights.W_K(input_key_x)
        value_x = self.att_weights.W_V(input_value_x)
        
        # split into multiple heads by reshaping or view => N x num_heads x L x d_model/num_heads
        query_x = query_x.reshape(N, L, self.num_heads, self.d_k).transpose(1,2)
        key_x = key_x.reshape(N, L, self.num_heads, self.d_k).transpose(1,2)
        value_x = value_x.reshape(N, L, self.num_heads, self.d_v).transpose(1,2)
       
        # get the scores by multiplying query by the key - dot product or attention of one query token w.r.t to other tokens as keys
        scores = query_x @ key_x.transpose(-2,-1)
        scores /= math.sqrt(self.d_k)
        
        if use_mask:
            print(f'Masking')
            mask = torch.tril(torch.ones(L,L)).view(1,1,L,L)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # apply softmax 
        prob_att = softmax(scores, dim=-1)
        y = prob_att @ value_x
        
        # reassemble all the heads
        y = y.transpose(1,2).contiguous().reshape(N, L, self.num_heads * self.d_v)
        y = self.dropout(self.proj_weights(y))
        
        return y
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p_dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        assert d_model <= 4 * d_ff
        self.d_model = d_model
        self.c_fc = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, x):
        return self.c_proj(self.dropout(self.activation(self.c_fc(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, att_module : nn.Module, ffn_module : nn.Module, p_dropout : float = 0.1):
        super(EncoderLayer, self).__init__()
        self.s_att = att_module
        self.ffn = ffn_module
        self.att_layernorm_conn = SubLayerConnection(self.s_att.d_model,p_dropout)
        self.ffn_layernorm_conn = SubLayerConnection(self.ffn.d_model, p_dropout)
    
    def forward(self, x, use_mask=False):
        x = self.att_layernorm_conn(x, lambda x : self.s_att(x,x,x, use_mask))
        return self.ffn_layernorm_conn(x, self.ffn)
    
class DecoderLayer(nn.Module):
    def __init__(self, att_module : nn.Module, ffn_module : nn.Module, p_dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.s_att = att_module
        self.cross_att = copy.deepcopy(att_module)
        self.ffn = ffn_module
        self.att_layernorm_conn = SubLayerConnection(self.s_att.d_model,p_dropout)
        self.cross_att_layernorm_conn = SubLayerConnection(self.cross_att.d_model,p_dropout)
        self.ffn_layernorm_conn = SubLayerConnection(self.ffn.d_model, p_dropout)
        
    def forward(self, x, memory, use_src_mask = False, use_tgt_mask = True):
        # first self attention on target will be masked
        x = self.att_layernorm_conn(x, lambda x : self.s_att(x,x,x,use_tgt_mask))
        
        # cross attention -using src mask since keys and values come from the encoder output
        x = self.cross_att_layernorm_conn(x, lambda x : self.cross_att(x, memory ,memory , use_src_mask))
        
        # feed forward
        x = self.ffn_layernorm_conn(x, self.ffn)
        return x
    
# define a encoder class which contains N layers (modules) and layer norm
class Encoder(nn.Module):
    def __init__(self, d_model : int, layer : nn.Module,  N : int):
        super(Encoder, self).__init__()
        self.norm = LayerNorm(d_model)
        self.layers = clones(layer, N)
        
    def forward(self, x, use_mask = False):
        """ 
        Pass input and mask through each layer
        """
        for layer in self.layers:
            x = layer(x, use_mask)
        return self.norm(x)
    
# decoder
class Decoder(nn.Module):
    def __init__(self, d_model : int, layer : nn.Module, N: int):
        super(Decoder, self).__init__()
        self.norm = LayerNorm(d_model)
        self.layers = clones(layer, N)
        
    def forward(self, x, memory, use_src_mask, use_tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, use_src_mask, use_tgt_mask)
        return self.norm(x)
    
# create the embedding layer
# the embedding layer weights are learned and shared between the input tokens and embedding, and output tokens and output embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    def forward(self,x):
        return self.lut(x) / (math.sqrt(self.d_model))
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 5000, p_dropout = 0.1):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p_dropout)
        
        # create the positional embedding 
        pos_term = torch.arange(0, max_seq_len).unsqueeze(1)
        c_term =  torch.exp( -1 * torch.arange(0, self.d_model, 2) * math.log(10000) )
        pe = torch.zeros(max_seq_len, self.d_model)
        pe[:,0::2] = torch.sin( pos_term * c_term)
        pe[:,1::2] = torch.cos( pos_term * c_term)
        pe = pe.unsqueeze(0) # to add the batch dimensioon
        
        # save this as buffer
        self.register_buffer("pe",pe)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)  
 
def test_multi_headed_attention():
    N = 20
    L = 10
    d_model = 512 # embedding dimensions
    input_x = torch.randn([N, L, d_model])
    m_att = MultiHeadedAttention(d_model=d_model, num_heads=8, p_dropout=0.1, d_k = 12, d_v=9)
    y = m_att(input_x, input_x, input_x, use_mask = True)
    print(y)
    
def test_pointwisefeedforward():
    d_model = 512
    d_ff = 2048
    N = 20
    L = 20
    
    ffn = PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    test_data = torch.randn([N, L, d_model])
    y = ffn(test_data)
    print(y)
    print(test_data)
    
def test_encoder_layer():
    d_model = 512
    d_ff = 2048
    N = 20
    L = 20
    num_heads = 8
    
    encoder_layer = EncoderLayer(
        att_module=MultiHeadedAttention(d_model=d_model, num_heads=num_heads, d_k=12, d_v=9),
        ffn_module=PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    
    test_data = torch.randn([N, L, d_model])
    y = encoder_layer(test_data, use_mask = False)
    #print(y) 
    
def test_decoder_layer():
    d_model = 512
    d_ff = 2048
    N = 20
    L = 20
    num_heads = 8
    
    encoder_layer = EncoderLayer(
        att_module=MultiHeadedAttention(d_model=d_model, num_heads=num_heads, d_k=12, d_v=9),
        ffn_module=PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    
    test_data = torch.randn([N, L, d_model])
    encoded_test_data = encoder_layer(test_data, use_mask = False) 
    
    decoder_layer = DecoderLayer(
        att_module= MultiHeadedAttention(d_model=d_model, num_heads=num_heads, d_k=12, d_v=9),
        ffn_module= PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    
    output_tgt_data = torch.randn([N, L, d_model])
    
    decoded_output_data = decoder_layer(output_tgt_data, encoded_test_data, use_src_mask = False, use_tgt_mask = True) 
    print(decoded_output_data)

def test_encoder_decoder():
    d_model = 512
    d_ff = 2048
    batch_size = 20
    L = 20
    num_heads = 8
    encoder_layer = EncoderLayer(
        att_module=MultiHeadedAttention(d_model=d_model, num_heads=num_heads, d_k=12, d_v=9),
        ffn_module=PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    decoder_layer = DecoderLayer(
        att_module= MultiHeadedAttention(d_model=d_model, num_heads=num_heads, d_k=12, d_v=9),
        ffn_module= PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    
    encoder = Encoder(d_model=d_model,layer = encoder_layer, N = 2)
    decoder = Decoder(d_model=d_model, layer= decoder_layer, N = 2)
    
    test_data = torch.randn([batch_size, L, d_model])
    output_tgt_data = torch.randn([batch_size, L, d_model]) 
    
    encoded_test_data = encoder(test_data)
    
    decoded_test_data = decoder(output_tgt_data, encoded_test_data, use_src_mask = False, use_tgt_mask = True)
    print(decoded_test_data)

def test_positional_encoding():
    d_model = 10
    seq_len = 100
    batch_size = 1
    pe = PositionalEmbedding(d_model=d_model)
    test_data = torch.zeros(batch_size,100,d_model)
    
    y = pe(test_data)
    print(y)

def create_model(src_vocab_size, tgt_vocab_size, d_model = 512, d_ff = 2048, N=6, num_heads = 8, p_dropout = 0.1):
    encoder_layer = EncoderLayer(
        att_module=MultiHeadedAttention(d_model=d_model, num_heads=num_heads),
        ffn_module=PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    decoder_layer = DecoderLayer(
        att_module= MultiHeadedAttention(d_model=d_model, num_heads=num_heads),
        ffn_module= PointWiseFeedForward(d_ff=d_ff, d_model=d_model)
    )
    
    encoder = Encoder(d_model=d_model,layer = encoder_layer, N = N)
    decoder = Decoder(d_model=d_model,layer = decoder_layer,  N = N) 
    
    position_embedder = PositionalEmbedding(d_model=d_model, p_dropout=p_dropout)
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab_size=src_vocab_size), 
                                                      copy.deepcopy(position_embedder)),
                              tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab_size=tgt_vocab_size), 
                                                      copy.deepcopy(position_embedder)),
                              generator=Generator(d_model=d_model, vocab_size=tgt_vocab_size))
    
    # initialize model parameters
    for p in transformer.parameters():
        if p.dim() > 1 : 
            nn.init.xavier_uniform_(p)
    return transformer
        
def test_transformer():
    
    orig_transformer = create_model(src_vocab_size=11, tgt_vocab_size=11, N=2)
    print(orig_transformer)
    
    orig_transformer.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    
    memory = orig_transformer.encode(src,use_mask=False)
    ys = torch.zeros(1, 1).type_as(src)
    
    # TODO: Ys is smaller than input. So, how do we handle that? 
    

    for i in range(9):
        out = orig_transformer.decode(memory=memory, use_src_mask=False, tgt=ys, use_tgt_mask=True).type_as(src.data)
        prob = orig_transformer.generator(out[:,-1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)
    
if __name__ == "__main__":
    #test_multi_headed_attention()
    #test_pointwisefeedforward()
    # test_decoder_layer()
    #test_encoder_decoder()
    #test_positional_encoding()
    test_transformer()
    