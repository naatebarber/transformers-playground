import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=torch.device("cpu")):
        super(MultiHeadAttention, self).__init__()

        # Make sure the model dimension is divisible by the number of heads.
        assert d_model % num_heads == 0, "d_model not divisible by num heads"

        self.device = device

        # Initialize dimensions
        self.d_model = d_model
        self.num_heads = num_heads

        """
        d_k represents the size of the key and query vectors in the attention mechanism.

        For example, if d_model (input space) is 1024 and we're splitting it over 8 attention heads,
        d_k would be 1024 / 8 -> 128.

        This is quite a high input space for each attention head, 
        which is why we normalize the attention score by dividing it by the sqrt of d_k

        attn_score = dp(q, k') / sqrt(d_k)
        """
        self.d_k = self.d_model // self.num_heads

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model).to(self.device)  # Query
        self.W_k = nn.Linear(d_model, d_model).to(self.device)  # Key
        self.W_v = nn.Linear(d_model, d_model).to(self.device)  # Value
        self.W_o = nn.Linear(d_model, d_model).to(self.device)  # Output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        """
        K and Q are tensors
        K and Q have shape represented by [batch_size, num_heads, seq_length, depth]

        When calculating attention scores we need to transpose K's last two dimensions
        K -> batch_size, num_heads, seq_length, depth
        K' -> batch_size, num_heads, depth, seq_length

        The matmul result will have shape:
        A -> batch_size, num_heads, seq_length, seq_length

        GPT {
            The result is a new tensor with shape
            [batch_size, num_heads, seq_length, seq_length]
            where each element [i, j] in the seq_length x seq_length matrix
            represents the dot-product (attention score) between the i-th query
            and j-th key for each head and each item in the batch.
        }

        / math.sqrt(self.d_k):

        Scale down the raw attention scores by the square root of d_k
         - This prevents the dot product from getting too large in scenarios where d_k is high.
         - If we dont normalize, each token may not receive its fair share of attention. For example:

             hello i am nate
        hello  1   0  0  0
        i      0   0  0  0
        am     0   0  0  0
        nate   0   0  0  0

        This would limit learning.
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        """
        Logic on whether or not to apply the mask. Let's break it down.

        This is useful for:
         - ignoring padding on the encode step
         - ensuring future tokens do not influence the current step. (causal masking, look-ahead masking)

        mask == 0 -> This creates a boolean tensor with True where the mask == 0. 
        Think of the mask as a tensor, this indicates the posititions where attention should be masked.

        attn_scores.masked_fill(...) -> This replaces all positions on the attn_scores tensor, where
        mask[...] == True, with the value -1e9 (-1 billion).

        This ensures that masked values will become negligible after softmax is applied, 
        since small values are pushed down and large values are enhanced.
        """
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        """
        We're running softmax over the last dimension of the attn_scores matrix.
        Remember, after our matmul, this cooresponds to sequence_length since our shape is:

        [batch_size, num_heads, seq_length, seq_length]

        This will reduce sum all values of attn_scores
        """
        attn_probs = torch.softmax(attn_scores, dim=-1)

        """
        Whereas K is used to determine what level of attention should be paid to the currently observed token Q,
        V holds the information we actually want to aggregate.

        Keys determine influence
        Values determine output (What is that output and how is it used?)

        The relationship between Q and K represents the importance of the current token Q within the sentence
        The relationship between the attention_probs and V represents the context of the current token Q within the sentence.
        """
        out = torch.matmul(attn_probs, V)
        return out

    def split_heads(self, x):
        # Reshape the input to have N heads for multi-head attention
        """
        Splitting heads essentially gives the model different sets of eyes to look at different parts of the input token vector.
        Say you have an input vector with 1024 features

        Specifically, this allows the model to search for features in a subset of an input space. Which is why multi-head attention is powerful.

        If you have four attention heads you're splitting over, each head will receive a 256 features in order.
        Head 1: 0-255
        head 2: 256-511
        head 3: 512-767
        head 4: 768-1023

        When the heads are rejoined, features between separate heads (such as feature 1 and feature 1023) will be coorelated. Wow.

        Also, this is not a direct split of the input data from X. Applying the linear
        transformations (W_q, W_k, W_v) will likely mix these features before they are fed to
        scaled_dot_product_attention
        """

        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine all the heads back into their original size.

        """
        x includes the output results of all the heads.
        what we are doing here is reshaping it back into it's original form.

        GPT {
            After processing the input data through multiple heads in parallel, each producing a
            part of the output, this method merges those parts back into a single tensor that
            has the same dimensionality as the original input before it was split among heads.
            This is achieved by reversing the operations that were applied to split the input,
            ensuring that the tensor's shape goes from [batch_size, num_heads, seq_length, d_k]
            back to [batch_size, seq_length, d_model], thus maintaining the original feature
            space dimensionality.
        }
        """
        batch_size, _, seq_length, d_k = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        # The linear transformation is performed first, and then the resulting tensor is split into multiple heads.
        # This allows each head to focus on different parts of the input vector, capturing different aspects of the information.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot product attention
        # The attention mechanism is applied to the queries, keys, and values.
        # The mask is used here if provided to prevent attention to certain positions, such as future positions in sequence or padding.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine the heads and apply the final linear transformation
        # After attention has been computed separately in each head, the outputs are concatenated and passed through an additional linear layer.
        # This final transformation combines information from all heads to produce the final output.
        output = self.W_o(self.combine_heads(attn_output))
        return output
