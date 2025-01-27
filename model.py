import warnings
from typing import Optional, List, Tuple
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertConfig, BertEmbeddings, BertEncoder, BertLayer, BertAttention
)
from transformers.modeling_utils import ModuleUtilsMixin
from torchmetrics import PearsonCorrCoef
from transformers import  BertPreTrainedModel
from einops import rearrange   
import numpy as np 
import random 
import torch.nn.functional as F

class soft_pool1d(nn.Module):
    def __init__(self,  kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride= kernel_size
    def forward(self, x):
        e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
        return F.avg_pool1d(x.mul(e_x), self.kernel_size, stride=self.stride).mul_(self.kernel_size).div_(F.avg_pool1d(e_x, self.kernel_size, stride=self.stride).mul_(self.kernel_size))


def lip1d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()

    return F.avg_pool1d(x*weight, kernel, stride, padding)/F.avg_pool1d(weight, kernel, stride, padding)
    
class LIP(nn.Module):
    def __init__(self, channels):
        super(LIP, self).__init__()
        rp = channels
        self.logit = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm1d(channels, affine=True),
                nn.ReLU(),
        )
    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip1d(x, self.logit(x))
        return frac

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2,dropout_prob=0.1):
        super().__init__()
        self.pool_size = pool_size
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.to_attn_logits1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_attn_logits2 = nn.ModuleList([nn.Conv1d(dim, dim, 1, bias=False) for _ in range(pool_size)])
        self.BN = nn.BatchNorm1d(dim)
    def forward(self, x):
        b, s, n = x.shape  
        remainder = n % self.pool_size
        needs_padding = remainder > 0      
        if needs_padding:
            x = F.pad(x, (0, (self.pool_size-remainder)), value=0) 
        x = x.unfold(-1,self.pool_size,self.pool_size) 
        outx = []
        i = 0
        for conv in self.to_attn_logits2:                
            nx = x[:,:,:,i]
            nx = self.BN(nx)
            logit = conv(nx)           
            outx.append(logit)
            i+=1 
        outx = torch.stack(outx, dim=-1)       
        logits =  self.to_attn_logits1(outx)
        logits = self.attn_dropout(logits)             
        attn = logits.softmax(dim=-1)             
        outs = (outx * attn).sum(dim=-1)
        return outs
  
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x
        
def ConvBlock(dim, dimout, kernel_size = 1,stride=2):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim,  dimout, kernel_size,stride=stride, padding = kernel_size // 2)
    )

class EBMGP(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        type_vocab_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_attention_heads: int = 8,
        intermediate_size: int = 256,
        hidden_act: str = "gelu",
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=5000, 
            dropout_rate = dropout_rate,
        )

        self.embeddings = BertEmbeddings(self.config)

        self.convs = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            ConvBlock(hidden_size, 64,30,stride=2),
            AttentionPool(64, pool_size = 1),
            #LIP(64),
            #nn.AvgPool1d(1),
            nn.Dropout(dropout_rate), 

            nn.BatchNorm1d(64),
            ConvBlock(64, 64, 3,stride=2),            
            AttentionPool(64, pool_size = 2),
            #nn.AvgPool1d(2),
            #LIP(64),
            nn.Dropout(dropout_rate), 

            nn.BatchNorm1d(64),
            ConvBlock(64, 64, 30,stride=2),           
            AttentionPool(64, pool_size = 3),
            #nn.AvgPool1d(3),
            #LIP(64),
            nn.Dropout(dropout_rate), 

            nn.BatchNorm1d(64),
            ConvBlock(64, 64, 3,stride=2),            
            AttentionPool(64, pool_size = 2),
            #nn.AvgPool1d(2),
            #LIP(64),
            nn.Dropout(dropout_rate), 

            nn.BatchNorm1d(64),
            ConvBlock(64, 64, 30,stride=2),           
            AttentionPool(64, pool_size = 3),
            #nn.AvgPool1d(3),
            #LIP(64),
            nn.Dropout(dropout_rate), 
        )
          
        self.predictor = nn.Sequential(           
            Rearrange('b n d-> b (n d)'), 
            nn.Linear(320,1), 
        ) 

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids.long(),
            position_ids=position_ids,
            token_type_ids=token_type_ids.long(),
        )
        x = embedding_output.permute(0, 2, 1) 
        x = self.convs(x)
        logits = self.predictor(x) 
        return logits
