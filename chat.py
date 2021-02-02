"""
File with functions implementing 'conversation'
with a trained model.
"""
import os
from typing import List

import yaml
import torch
import torch.nn as nn

import models
from utils.vocabulary import Vocabulary
from utils.preprocessing import basic_preprocessing


def evaluate(greedy_searcher: models.GreedySearchDecoder,
             vocabulary: 'Vocabulary',
             sentence: str,
             max_length: int,
             sos_token: int,
             device: torch.device) -> List[str]:
    """
    Function given a sentence return a most probable next sentence.

    Parameters
    --------
    greedy_searcher: models.GreedySearchDecoder
        Models performing greedy decoding.
    vocabulary: Vocabulary
        Vocabulary providing model with a word to index mapping.
    sentence: str
        Sentence for which we are finding a successor.
    max_length: int
        Maximum sentence length.
    sos_token: int
        Start of a sentence token.
    device: torch.device
        Device on which calculations are performed.
    Returns
    -------
    List[str]
        Most probable following sentence after passed one.
    """
    index_batch: List[List[int]] = [vocabulary.index_from_sentence(sentence)]
    print(index_batch)
    lengths: torch.Tensor =\
        torch.Tensor([len(index) for index in index_batch]).to(device)
    input_batch = torch.LongTensor(index_batch).transpose(0, 1).to(device)
    tokens, scores = greedy_searcher(input_batch, lengths, max_length, sos_token, device)
    return [vocabulary.index2word[token.item()] for token in tokens]


def chat(greedy_searcher: models.GreedySearchDecoder,
         vocabulary: 'Vocabulary',
         max_length: int,
         sos_token: int,
         device: torch.device) -> None:
    while True:
        try:
            input_seq: str = basic_preprocessing(input("> "))
            if input_seq == "q":
                break
            out: List[str] = evaluate(
                greedy_searcher=greedy_searcher,
                vocabulary=vocabulary,
                sentence=input_seq,
                max_length=max_length,
                sos_token=sos_token,
                device=device
            )
            out = [word for word in out
                    if word != "EOS" and word != "PAD"]
            print(f"Bot: {' '.join(out)}")
        except KeyError:
            print("Word not known.")


if __name__ == "__main__":
    with open("settings/config.yaml", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error during config load: {str(e)}")

    try:
        checkpoint: dict = torch.load(
                os.path.join(config["data"]["save_dir"], "model.tar")
        )
    except FileNotFoundError as e:
        print(f"Train model before chatting!")

    vocabulary: 'Vocabulary' = Vocabulary([], config)
    ocabulary.__dict__ = checkpoint["vocabulary"]

    device: torch.device =\
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size: int = config['training']['hidden_size']
    dropout_ratio: float = config['training']['dropout_ratio']

    embedding: nn.Embedding = \
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

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    greedy_searcher: models.GreedySearchDecoder =\
        models.GreedySearchDecoder(encoder, decoder)
    encoder.eval()
    decoder.eval()
    greedy_searcher.eval()

    chat(
        greedy_searcher=greedy_searcher,
        vocabulary=vocabulary,
        max_length=config["vocabulary"]["max_length"],
        sos_token=config["vocabulary"]["sos_token"],
        device=device
    )
