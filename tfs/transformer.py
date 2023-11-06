import torch
import torch.nn as nn

from tfs.positional_encoding import PositionalEncoding
from tfs.encoder_layer import EncoderLayer
from tfs.decoder_layer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        """
        src_vocab_size = the number of individual input tokens the encoder can handle
        target_vocab_size = the number of individual output tokens the decoder can write
        """


        """
        The embeddings defined below will create src_vocab_size rows, each with d_model columns. 

        The embeddings defined below will create an embedding matrix for both the encoder and decoder.
        The encoder embedding matrix will have 'src_vocab_size' rows, and the decoder embedding matrix
        will have 'target_vocab_size' rows. The number of columns in each matrix should match 'd_model',
        which is the dimensionality of the Transformer model's hidden layers. These embedding matrices
        map each token index to a high-dimensional space and are learnable parameters that the model
        will adjust during training. These embeddings are crucial for the model's ability to understand
        and generate language, as they provide the basis for the self-attention mechanisms to compute
        contextual relationships between words.
        """
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        This creates a boolean mask where all elements of the src tensor 
        that are not equal to 0 are marked as True (meaning they are valid tokens), 
        and all 0s (typically padding tokens) are marked as False. This assumes 
        that the index 0 is used for padding.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        """
        Get the sequence length of the target tensor (decoder layer)
        """
        seq_length = tgt.size(1)

        """
        Create a square 2D tensor of ones with dimensions [seq_length, seq_length]. This represents a mask
        where initially all positions can 'see' each other. A dimension of size 1 is prepended to allow this mask
        to be easily broadcasted across the batch during batch-wise operations.
        """
        ones = torch.ones(1, seq_length, seq_length)

        """
        Apply the torch.triu (upper triangular) function to generate a mask with ones above the main diagonal
        (starting from 1 above the main diagonal) and zeros on and below the diagonal. The 'diagonal=1' argument
        means the diagonal of the matrix (representing each token's self-attention) is set to zero and is not included
        in the 'no peek' area.

        Subtracting this from 1 inverts the mask: the upper triangular part (excluding the main diagonal) now contains zeros,
        indicating positions that the model should not attend to (future tokens), and the lower triangular part including the 
        diagonal contains ones, indicating positions that are allowed (past tokens and the token itself).

        Casting to a Boolean tensor is the final step. In PyTorch, True values are treated as 'masked' and therefore ignored 
        in computations such as attention, while False values are treated as 'unmasked' and included in computations. Thus,
        the resulting mask prevents the decoder from 'peeking' into the future tokens during the self-attention calculations.
        """
        nopeak_mask = (1 - torch.triu(ones, diagonal=1)).bool()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Generate masks using the above method
        """
        src_mask, target_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer.forward(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer.forward(dec_layer, enc_output, src_mask, target_mask)

        output = self.fc(dec_output)
        return output