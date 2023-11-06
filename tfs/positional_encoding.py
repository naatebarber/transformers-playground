import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()

        self.device = device

        """
        Initialize a tensor of zeros that will hold the positional encodings.
        The tensor's dimensions are determined by the maximum sequence length
        (the length of the longest sequence or the maximum number of tokens)
        and the dimensionality of the model (d_model).
        """
        pe = torch.zeros(max_seq_length, d_model).to(self.device)

        """
        torch.arange -> Generates a 1D tensor containing a range from 0 up to
        but not including the max sequence length, where each value represents a
        position in the sequence. The resulting tensor is a series of floats
        starting at 0.0 and increasing by 1.0 up to max_seq_length - 1.

        For a max_seq_length of 3, the tensor would be: [0.0, 1.0, 2.0].

        .unsqueeze(1) -> Adds an additional dimension at the specified index (1).
        This transforms the 1D tensor into a 2D tensor where the original values
        become rows, allowing us to combine these positions with the encoding
        tensor through broadcasting.

        Thus, after unsqueezing at index 1, the tensor changes shape from [max_seq_len]
        to [max_seq_len, 1], becoming: [[0.0], [1.0], [2.0]].

        If .unsqueeze(0) were used instead, it would transform the tensor to
        [1, max_seq_len], like this: [[0.0, 1.0, 2.0]].

        This position tensor is a crucial component for incorporating sequence
        information into the positional encodings, which is necessary since the
        self-attention mechanism in the Transformer model does not inherently
        capture the order of the input sequence.
        """
        position = (
            torch.arange(0, max_seq_length, dtype=torch.float)
            .unsqueeze(1)
            .to(self.device)
        )

        """
        frequency_step is a scaling factor used to determine how rapidly the sine and cosine waves oscillate. 
        It uses a logarithmic scale based on the model's dimensions, d_model. 
        This value is made negative because the exponent will be used to create a decay factor in div_term.

        div_term is a term used to apply the frequency decay for each dimension of the positional encoding. 
        The torch.arange(0, d_model) generates a tensor with values from 0 to d_model - 1, 
        which, when multiplied by the frequency_step and exponentiated, gives a series of values that decay on a log scale. 
        This is because as i increases, -(math.log(10000) / d_model) * i becomes more negative, so exp of this becomes smaller.
        """

        # Creates a spectrum of wavelengths
        frequency_step = -(math.log(10000)) / d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * frequency_step).to(
            self.device
        )

        """
        pe[:, 0::2] = torch.sin(position * div_term) assigns the sine of the positional 
        encoding to even indices of the encoding matrix (pe). 
        The position variable is broadcasted across the second dimension, 
        which allows for each position to have a unique sinusoidal encoding based on its value.
        """
        pe[:, 0::2] = torch.sin(position * div_term)

        """
        pe[:, 1::2] = torch.cos(position * div_term) does the same for the odd indices, 
        but with the cosine function.
        """
        pe[:, 1::2] = torch.cos(position * div_term)

        """
        This line is registering the pe tensor as a buffer within the module. 
        By unsqueezing pe at the first dimension (unsqueeze(0)), 
        the method adds a batch dimension with size 1 at the beginning, 
        which allows the positional encodings to be easily broadcasted and 
        added to the batch of embeddings during forward passes through the network.
        """

        """
        pe.unsqueeze(0) -> Before this operation, pe has a shape of [max_seq_len, d_model]
        representing one position encoding for every dimension of d_model.
        After unsqueeze, the new shape is [1, max_seq_len, d_model] allowing the positional encodings
        to be broadcast to every token in the input_sequence.
        """
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds positional encodings to the input tensor x using broadcasting.

        The positional encoding tensor self.pe is sliced up to the sequence length
        of the input tensor x to ensure matching dimensions for element-wise addition.
        This broadcasting step is implicitly handled by PyTorch and allows the positional
        encodings to be added to each sequence in the batch, effectively incorporating
        position information into the token embeddings.

        Parameters:
        - x: A tensor of shape [batch_size, seq_length, d_model] representing the input token embeddings.

        Returns:
        - A tensor of the same shape as x with positional encodings added to the token embeddings.
        """
        return x + self.pe[:, : x.size(1)]
