import torch
import torch.nn as nn
import torch.utils as utils

from tfs.multi_head_attention import MultiHeadAttention
from tfs.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

        """
        Layer normalization is a technique to normalize the values across each feature in a layer's output. 
        Unlike batch normalization, which computes statistics across the batch dimension, layer normalization 
        computes statistics across the feature dimension (i.e., it normalizes the values for each feature within 
        a single example independently).

        In the context of a Transformer's encoder, each sublayer (self-attention and feed-forward) outputs a set 
        of features for each token in the sequence. The outputs can vary widely in scale and variance. Layer normalization 
        helps to stabilize the training process by ensuring that the outputs of these sublayers have a mean of zero 
        and a standard deviation of one across the features. This is especially important in deep networks, like Transformers, 
        where small changes in the lower layers can be amplified as the data moves through the network, potentially 
        leading to issues like exploding or vanishing gradients.

        In practice, layer normalization works by subtracting the mean and dividing by the standard deviation for each 
        feature. Parameters (a scale and a shift) are then learned for each feature, which allows the network to 
        undo the normalization if that is beneficial (this is akin to how batch normalization has learnable gamma 
        and beta parameters).

        In the nn.LayerNorm module, the `d_model` parameter specifies the number of features to normalize across. 
        Applying this normalization helps to ensure that each feature contributes approximately equally to the subsequent 
        computations and can help the model to learn and converge faster.

        In the Transformer architecture, layer normalization is typically applied after the residual connection (the 
        'add' step in 'add-and-norm'), allowing the model to benefit from both the raw input and the stabilized output 
        of each sublayer.
        """
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        """
        The dropout parameter provided when initializing the EncoderLayer class 
        determines the probability at which each element is dropped. This module is 
        typically used in conjunction with the "add-and-norm" steps in a Transformer, where 
        it is applied after each sublayer's output and before it is added to the sublayer's input 
        (forming a residual connection) and normalized.
        """
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        In the encoder stage, Q, K, and V are all the same.
        An attention mask may be applied.
        """
        attn_output = self.self_attn(x, x, x, mask)

        """
        Here we normalize and apply dropout to the output of the self_attention step.
        Dropout zeros some of the weights from attn_output, preventing overfitting.

        The dropout-modified self-attention output is then added 
        to the original input x through a residual connection: x + self.dropout(attn_output). 
        The idea is to carry forward the original information, which ensures that the network can still 
        perform at least as well as not having these additional layers. Residual connections help with the 
        flow of gradients during backpropagation, especially in deep networks, by providing alternative pathways.
        """
        x = self.norm_1(x + self.dropout(attn_output))

        """
        We do the same thing in the feed_forward step
        """

        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout(ff_output))
        return x
