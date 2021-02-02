"""
File with definitions of encoder and decoder.
"""
import enum
from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionType(enum.Enum):
    """
    Enum of all attention types.
    """
    DOT = "dot"
    GENERAL = "general"
    CONCAT = "concat"


class Encoder(nn.Module):
    """
    Class implementing multi-layered bidirectional GRU encoder.

    Attributes
    ----------
    hidden_size: int
        Size of hidden layers dimension.
    embedding: torch.nn.Embedding
        Used to encode words indices iin an arbitrarily sized feature space.
    num_layers: int
        Number of layers in GRU module.
    """
    def __init__(self, hidden_size: int, embedding: nn.Embedding,
                 num_layers: int, dropout: float) -> None:
        """
        Parameters
        ----------
        hidden_size: int
            Size of hidden layers dimension.
        embedding: torch.nn.Embedding.
            Torch embedding module.
        num_layers: int
            Number of GRU's layers.
        dropout: float
            Determines if dropout should be used.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = embedding
        self.num_layers: int = num_layers

        self.gru: nn.GRU = nn.GRU(self.hidden_size, self.hidden_size,
                                  num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seq: torch.Tensor,
                input_lengths: List[int], hidden: Union[None, torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method implementing encoder's forward pass.

        Parameters
        ----------
        input_seq: torch.Tensor
            Batch of input sequences.
        input_lengths: List[int]
            List of sequences lengths corresponding to each sentence
            in the batch.
        hidden: Union[None, torch.Tensor], optional
            Hidden state (initially set to None).
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of calculated output features in the last GRU's layer and
            updated hidden state.
        """
        embedded: torch.Tensor = self.embedding(input_seq)
        packed: nn.utils.rnn.PackedSequence =\
            nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        out, hidden = self.gru(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # sum bidirectional GRU outputs
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        return out, hidden


class Attention(nn.Module):
    """
    Implementation of a global attention layer.

    Attributes
    ----------
    method: str
        Determines how attention is calculated
    hidden_size: int
        Size of hidden state.
    attention: nn.Linear
        Linear layer representing attention.
    v: nn.Parameter, optional
        Tensor used in one of the attention calculation method.
    """
    def __init__(self, method: str, hidden_size: int) -> None:
        """
        Parameters
        ----------
        method: str
            Method how attention is calculated.
        hidden_size: int
            Size of hidden state.
        """
        super().__init__()
        self.method: str = method
        self.hidden_size: int = hidden_size

        supported_attention_methods: List[str] =\
            [e.value for e in AttentionType]
        if self.method not in supported_attention_methods:
            raise ValueError("Supported attention calculation methods are"
                             f"{supported_attention_methods}")

        if self.method == AttentionType.GENERAL.value:
            self.attention: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == AttentionType.CONCAT.value:
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v: nn.Parameter = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def general(self, hidden: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention using general method.

        Parameters
        ----------
        hidden: torch.Tensor
            Calculated hidden state.
        encoder_out: torch.Tensor
            Passed encoder output.
        Returns
        -------
        torch.Tensor
            Calculated attention.
        """
        return torch.sum(hidden * self.attention(encoder_out), dim=2)

    def concat(self, hidden: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention using concat method.

        Parameters
        ----------
        hidden: torch.Tensor
            Calculated hidden state.
        encoder_out: torch.Tensor
            Passed encoder output.
        Returns
        -------
        torch.Tensor
            Calculated attention.
        """
        energy: torch.Tensor = self.attention(
            torch.cat((hidden.expand(encoder_out.size(0), -1, -1), encoder_out, 2)).tanh()
        )
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Implements forward pass of the module.
        Calculates attention using method passed in the constructor.

        Parameters
        ----------
        hidden: torch.Tensor
            Calculated hidden state.
        encoder_out: torch.Tensor
            Passed encoder output.
        Returns
        -------
        torch.Tensor
            Calculated attention score.
        """
        if self.method == AttentionType.DOT.value:
            attention: torch.Tensor = Attention.dot(hidden, encoder_out)
        elif self.method == AttentionType.GENERAL.value:
            attention = self.general(hidden, encoder_out)
        elif self.method == AttentionType.CONCAT.value:
            attention = self.concat(hidden, encoder_out)

        return F.softmax(attention.t(), dim=1).unsqueeze(1)

    @staticmethod
    def dot(hidden: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention using dot method.

        Parameters
        ----------
        hidden: torch.Tensor
            Calculated hidden state.
        encoder_out: torch.Tensor
            Passed encoder output.
        Returns
        -------
        torch.Tensor
            Calculated attention.
        """
        return torch.sum(hidden * encoder_out, dim=2)


class Decoder(nn.Module):
    """
    Class implementing multi-layered bidirectional GRU
    and attention decoder.

    Attributes
    ----------
    attention: Attention
        Attention layer.
    embedding: torch.nn.Embedding
        Used to encode words indices iin an arbitrarily sized feature space.
    hidden_size: int
        Size of hidden state.
    out_size: int
        Size of calculated output.
    num_layers: int
        Number of layers in GRU unit.
    dropout: float
        Dropout ratio used in GRU.
    embedding_dropout: nn.Dropout
        Dropout used after embedding.
    gru: nn.GRU
        Used GRU module.
    concat: nn.Linear
        Layer which connects concated layer to hidden size.
    out: nn.Linear
        Output Layer of the model.
    """
    def __init__(self, attention_method: AttentionType, embedding: nn.Embedding,
                 hidden_size: int, out_size: int, num_layers: int, dropout: float
                 ) -> None:
        """
        Parameters
        attention_method: AttentionType
            Method used to calculate attention.
        embedding: nn.Embedding
            Embedding layer.
        hidden_size: int
            Size of hidden state.
        out_size: int
            Size of output.
        num_layers: int
            Number of layers in GRU.
        dropout: float
            Dropout ratio used in GRU.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.out_size: int = out_size
        self.num_layers: int = num_layers
        self.dropout: float = dropout

        self.attention = Attention(attention_method, hidden_size)
        self.embedding: nn.Embedding = embedding
        self.embedding_dropout: nn.Dropout = nn.Dropout(dropout)
        self.gru: nn.GRU = nn.GRU(hidden_size, hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, input_step: torch.Tensor,
                hidden: torch.Tensor,
                encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method implements forward pass of a decoder.

        Parameters
        ----------
        input_step: torch.Tensor
            One time step of input sequence batch.
        hidden: torch.Tensor
            Final hidden layer of GRU.
        encoder_out: torch.Tensor
            Encoder's output.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
             Tuple of softmax normalized tensor giving probabilities of
             each word being correct next word in the decoded sequence and
             final hidden state of GRU.
        """
        embedded: torch.Tensor = self.embedding_dropout(
            self.embedding(input_step)
        )
        gru_out, hidden = self.gru(embedded, hidden)
        attn: torch.Tensor = self.attention(gru_out, encoder_out)

        # multiply attention weights to encoder outputs
        context: torch.Tensor = attn.bmm(encoder_out.transpose(0, 1))
        concat_input: torch.Tensor = torch.cat(
            (gru_out.squeeze(0), context.squeeze(1)), 1
        )
        concat_out: torch.Tensor = torch.tanh(
            self.concat(concat_input)
        )
        return F.softmax(self.out(concat_out), dim=1), hidden


class GreedySearchDecoder(nn.Module):
    """
    Class implementing greedy decoding.
    Word is chosen from decoder_output with the highest softmax value.
    Decoding is optimal on a single-step level.

    Attributes
    ----------
    encoder: Encoder
        Trained encoder model.
    decoder: Decoder
        Trained decoder model.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """
        Parameters
        ----------
        encoder: Encoder
            Implemented encoder model.
        decoder: Decoder
            Implemented decoder model.
        """
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(self, input_seq: torch.Tensor,
                input_length: torch.Tensor,
                max_length: int,
                sos_token: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation of forward pass.

        Parameters
        ----------
        input_seq: torch.Tensor
            Tensor of input sequences.
        input_length: torch.Tensor
            Lengths of sequences.
        max_length: int
            Max sentence length.
        sos_token: int
            Start of the sentence token.
        device: torch.device
            Device on which calculation will be performed.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of torch tensor which contains all tokens and scores
            returned by decoder.
        """
        encoder_out, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        decoder_in: torch.Tensor =\
            sos_token * torch.ones(1, 1, device=device, dtype=torch.long)
        all_tokens: torch.Tensor =\
            torch.zeros([0], device=device, dtype=torch.long)
        all_scores: torch.Tensor =\
            torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_out, decoder_hidden =\
                self.decoder(decoder_in, decoder_hidden, encoder_out)
            decoder_scores, decoder_in = torch.max(decoder_out, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_in), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_in = torch.unsqueeze(decoder_in, 0)
        return all_tokens, all_scores
