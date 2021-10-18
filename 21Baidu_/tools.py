import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from torch import nn

from torch import Tensor, Size
from typing import Union, List
_shape_t = Union[int, List[int], Size]

class ConditionalLayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, conditional_size: int ,eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.conditional_size = conditional_size
        self.weight_dense = nn.Linear(conditional_size,self.normalized_shape[0],bias=False)
        self.bias_dense = nn.Linear(conditional_size, self.normalized_shape[0],bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
            init.zeros_(self.weight_dense.weight)
            init.zeros_(self.bias_dense.weight) 

    def forward(self, input: Tensor,conditional: Tensor) -> Tensor:
        conditional = torch.unsqueeze(conditional, 1)
        add_weight =  self.weight_dense(conditional)
        add_bias = self.bias_dense(conditional)
        weight = self.weight + add_weight
        bias = self.bias + add_bias
        outputs = input
        mean = torch.mean(outputs, dim=-1, keepdim=True)
        outputs = outputs - mean
        variance = torch.mean(torch.square(outputs), dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        outputs = outputs / std
        outputs = outputs * weight
        outputs = outputs + bias
        return outputs

class GlobalAveragePooling1D(Module):
    def __init__(self,support_mask=True) -> None:
        super(GlobalAveragePooling1D, self).__init__()
        self.supports_masking = True
    
    def forward(self, inputs,mask=None,dim=1):
        if self.supports_masking:
            mask = mask.type(inputs.dtype)
            mask = mask.unsqueeze(2)
            inputs *= mask
            return torch.sum(inputs, dim=dim) / torch.sum(mask,dim=dim)
        else:
            return torch.mean(inputs, dim=dim)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
