"""
File implementing training loop of a defined architecture.
"""
import random
import logging
from typing import List

import yaml
import numpy as np
import torch
import torch.nn as nn

import models
#import test as models
from utils.loader import Loader
from utils.loss_fn import mask_nll_loss


def configure_device_for_optimizer(opt: torch.optim.Optimizer,
                                   device: torch.device
                                   ) -> torch.optim.Optimizer:
    """
    Configure optimizer to work on a given device.

    Parameters
    ----------
    opt: torch.optim.Optimizer
        Optimizer instance to be configured.
    device: torch.device
        Device on which optimizer will be working.
    Returns
    -------
    torch.optim.Optimizer
        Optimizer configured on a given device.
    """
    for state in opt.state.values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                state[key] = val.to(device)

    return opt


def train_step(input_var: torch.LongTensor,
               lengths: torch.Tensor,
               target_var: torch.LongTensor,
               mask: torch.BoolTensor,
               max_target_len: int,
               batch_size: int,
               clip: float,
               teacher_forcing_ratio: float,
               sos_token: int,
               encoder: models.Encoder,
               decoder: models.Decoder,
               encoder_opt: torch.optim.Optimizer,
               decoder_opt: torch.optim.Optimizer,
               device: torch.device
               ):
    """
    Function implementing single training step of a model.

    Parameters
    ----------
    input_var: torch.LongTensor
        Tensor of inputs.
    lengths: torch.Tensor
        Tensor of lengths of input sequences.
    target_var: torch.LongTensor
        Tensor of target values.
    mask: torch.Tensor
        Binary tensor describing the padding od the target tensor.
    max_target_len: int
        Maximum target length.
    batch_size: int
        Size of a batch.
    clip: float
        Threshold used for gradient clipping.
    teacher_forcing_ratio: float
        Threshold used for teacher forcing.
    sos_token: int
        Start of a sentence token.
    encoder: models.Encoder
        Encoder module.
    decoder: models.Decoder
        Decoder module.
    encoder_opt: torch.optim.Optimizer
        Optimizer used for encoder.
    decoder_opt: torch.optim.Optimizer
        Optimizer used for decoder
    device: torch.device
        Device on which calculation will be performed
    Returns
    -------
    float
        Loss calculated during performing
        training step.
    """
    loss: float = 0.
    totals: int = 0.
    loss_seq: List[float] = []
    input_var = input_var.to(device)
    target_var = target_var.to(device)
    mask = mask.to(device)
    lengths = lengths.to('cpu')

    encoder.zero_grad()
    decoder.zero_grad()

    encoder_out, encoder_hidden = encoder(input_var, lengths)
    decoder_input: torch.LongTensor = torch.LongTensor([
        [sos_token] * batch_size
    ]).to(device)
    # initial value to encoder final state
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    if random.random() < teacher_forcing_ratio:
        # teacher forcing - next input is current target
        for t in range(max_target_len):
            decoder_out, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_out
            )
            decoder_input = target_var[t].view(1, -1)
            mask_loss, total = mask_nll_loss(decoder_out,
                                             target_var[t], mask[t])
            loss += mask_loss
            totals += total
            loss_seq.append(mask_loss.item() * totals)
    else:
        for t in range(max_target_len):
            decoder_out, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_out
            )
            _, top_i = decoder_out.topk(1)
            decoder_input = torch.LongTensor(
                [[top_i[i][0] for i in range(batch_size)]]
            ).to(device)
            mask_loss, total = mask_nll_loss(decoder_out,
                                             target_var[t], mask[t])
            loss += mask_loss
            totals += total
            loss_seq.append(mask_loss.item() * totals)

    loss.backward()

    # clip gradients to max value
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_opt.step()
    decoder_opt.step()
    return sum(loss_seq) / totals


def train(vocabulary: 'Vocabulary',
          config: dict) -> None:
    """
    Function implementing training loop.

    Parameters
    ----------
    vocabulary: Vocabulary
        Vocabulary instance providing data for training.
    config: dict
        Dictionary containing parameters of a training loop.
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available()
                                        else 'cpu')

    hidden_size: int = config["training"]["hidden_size"]
    dropout_ratio: float = config["training"]["dropout_ratio"]

    embedding: nn.Embeding = \
        nn.Embedding(vocabulary.num_words, hidden_size).to(device)
    encoder: models.Encoder = \
        models.Encoder(hidden_size, embedding,
                       config["training"]["encoder_gru_layers"],
                       dropout_ratio).to(device)
    decoder: models.Decoder = \
        models.Decoder(config["training"]["attention_method"],
                       embedding, hidden_size, vocabulary.num_words,
                       config["training"]["decoder_gru_layers"],
                       dropout_ratio).to(device)

    encoder_opt: torch.optim.Optimizer = \
        torch.optim.Adam(encoder.parameters(),
                         lr=config["training"]["encoder_learning_rate"])
    decoder_opt: torch.optim.Optimizer = \
        torch.optim.Adam(decoder.parameters(),
                         lr=config["training"]["decoder_learning_rate"])
    encoder_opt = configure_device_for_optimizer(encoder_opt, device)
    decoder_opt = configure_device_for_optimizer(decoder_opt, device)

    training_batches: list = [
       vocabulary.batch_to_train_data(
           [random.choice(vocabulary.pairs)
            for _ in range(config["training"]["batch_size"])])
       for _ in range(config["training"]["epochs"])
    ]
    print("Training initialized.")

    total_loss: float = 0.0
    for epoch, training_batch in enumerate(training_batches, start=1):
        input_var, lengths, target_var, mask, max_target_len = training_batch
        loss: float = train_step(
            input_var=input_var,
            lengths=lengths,
            target_var=target_var,
            mask=mask,
            max_target_len=max_target_len,
            batch_size=config["training"]["batch_size"],
            clip=config["training"]["clip"],
            teacher_forcing_ratio=config["training"]["teacher_forcing_ratio"],
            sos_token=config["vocabulary"]["sos_token"],
            encoder=encoder,
            decoder=decoder,
            encoder_opt=encoder_opt,
            decoder_opt=decoder_opt,
            device=device
        )
        total_loss += loss
        if epoch % config["training"]["log_freq"] == 0:
            average_loss: float = total_loss / config["training"]["log_freq"]
            print(f"Epoch: {epoch:2}| Average loss: {average_loss:.3f}")
            total_loss = 0.0

        if epoch % config["training"]["save_freq"] == 0:
            pass


if __name__ == "__main__":
    with open("settings/config.yaml", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error during config load: {str(e)}")

    loader: Loader = Loader()
    loader.format_movie_lines()
    vocabulary: 'Vocabulary' = loader.load_prepared_data()
    train(vocabulary, config)
