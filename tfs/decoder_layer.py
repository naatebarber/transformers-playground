import torch
import torch.nn as nn

from tfs.multi_head_attention import MultiHeadAttention
from tfs.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        """
        Similar to encoder layer
        """

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, target_mask):

        """
        The mask here ensures the self-attention submodel can only create predictions for a token at i
        based on tokens of index < i.

        This prevents the attention submodel from looking at future tokens, ensuring causality.
        """
        attn_output = self.self_attn(x, x, x, target_mask)
        x = self.norm_1(x + self.dropout(attn_output))

        # """
        # In cross-attention, the queries come from the output of the previous self-attention layer 
        # of the decoder, while the keys and values are from the output of the encoder. This is essential for allowing 
        # the decoder to consider the entire input sequence when generating the output.

        # src_mask -> prevents the decoder from paying attention to padding tokens or other unwanted sources of input

        # The parameter x, in this case, represents the decoders own output, which it builds on autoregressively.

        # So first, the decoder maps attention on what it already said (x - its own output)

        # Then, the decoder takes the normalized result of self-attention, and uses it as the Q value to get
        # context from the encoder - relating what it just said to it's understanding of the input sequence.
        # """

        """
        In the cross-attention step, the queries come from the previous layer's output in the decoder, 
        and the keys and values come from the final output of the encoder. This cross-attention mechanism allows 
        the decoder to focus on different parts of the input sequence as needed to generate each element 
        of the output sequence.

        The src_mask is applied to prevent the decoder from attending to padding tokens or other irrelevant parts of the input,
        ensuring that only meaningful content influences the generation process.

        The variable 'x' represents the sequence generated by the decoder up to the current point, which is refined iteratively.
        The decoder uses its current state (x) to query the encoder's output and align its generated sequence accordingly.

        First, the decoder refines 'x' using self-attention, allowing it to consider what has been generated so far.
        Then, it employs cross-attention to incorporate context from the encoder, effectively aligning its own outputs with the input sequence.
        """
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm_2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm_3(x + self.dropout(ff_output))

        return x


