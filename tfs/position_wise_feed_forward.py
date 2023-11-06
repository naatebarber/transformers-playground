import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()

        """
        Define a non-linear feed forward network with a hidden layer size of d_ff

        d_ff should be larger than d_model to capture more complex patterns.
        PositionWise implies that this feed forward network will be applied to each position (each token in sequence length)
        after the self-attention heads have been reassembled.

        Self attention heads look at all positions and get context within their chunk of features (d_k length).
        These subspaces are combined back into a feature space of d_model
        Each position (token in sequence length) is fed through this network to introduce non-linearity.

        The intent is to introduce non-linearity to the system.

        GPT {
            Define a non-linear feed-forward network with a hidden layer size of d_ff.
            d_ff is typically larger than d_model, allowing the network to potentially capture more complex patterns. 
            This configuration can also be seen as a way to expand the representation before applying a non-linearity and then project it back to the original d_model dimension.
            The term 'PositionWise' means that this feed-forward network is applied to each position independently and identically. 
            This occurs after self-attention operations, where the multiple heads are already combined back into a single d_model-sized representation for each token.
            In the context of a Transformer, each token across the sequence length is passed through this feed-forward network. 
            The self-attention mechanism computes contextual relationships between tokens, and this network is applied to each resulting token vector in isolation, which introduces necessary non-linearity. 
            This process enables the model to learn more complex functions than what would be possible with self-attention alone.
            The inclusion of non-linearity is essential for deep learning models as it allows them to approximate more complex functions and interactions between input features.
        }
        """

        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc_2(self.relu(self.fc_1(x)))
