"""
File contains vocabulary class which contains mapping from words
to indexes and reverse mapping, count of each word and total number of
words.
"""
from itertools import zip_longest
from typing import Callable, Dict, List, Tuple

import torch

Pairs = List[List[str]]


class Vocabulary:
    PAD_token: int = 0
    SOS_token: int = 1
    EOS_token: int = 2

    def __init__(self, name: str, pairs: Pairs, *,
                 min_count_to_trim: int = 3) -> None:
        self.name: str = name
        self.pairs = pairs
        self.min_count_to_trim: int = min_count_to_trim
        self.trimmed: bool = False
        self.num_words: int = 3
        self.word2index: dict = {}
        self.word2count: dict = {}
        self.index2word: Dict[int, str] = {
            Vocabulary.PAD_token: "PAD",
            Vocabulary.SOS_token: "SOS",
            Vocabulary.EOS_token: "EOS"
        }
        self.filter_all_pairs()
        self.add_pairs()
        self.trim_rare_words()

    def add_pairs(self) -> None:
        for pair in self.pairs:
            for sentence in pair:
                for word in sentence.split():
                    self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def filter_all_pairs(self) -> None:
        self.pairs = [pair for pair in self.pairs
                      if Vocabulary.filter_pair(pair)]

    def trim(self) -> None:
        if self.trimmed:
            return
        self.trimmed = True
        keep_words: List[str] = [
            k for k, v in self.word2count.items()
            if v >= self.min_count_to_trim
        ]

        #print(f"keep_words {len(keep_words)}/{len(self.word2index)} = "
        #      f"{len(keep_words) / len(self.word2count):.4f}")

        self.num_words = 3
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            Vocabulary.PAD_token: "PAD",
            Vocabulary.SOS_token: "SOS",
            Vocabulary.EOS_token: "EOS"
        }

        for word in keep_words:
            self.add_word(word)

    def trim_rare_words(self) -> None:
        self.trim()

        keep_sentence: Callable[[str], bool] = \
            lambda sentence: all(word in self.word2index for word in sentence.split())
        keep_pairs: Pairs = [
            pair for pair in self.pairs
            if all(keep_sentence(sentence) for sentence in pair)
        ]

        #print("Trimmed from {} pairs to {}, {:.4f} of total" \
        #      .format(len(self.pairs), len(keep_pairs), len(keep_pairs) / len(self.pairs)))
        self.pairs = keep_pairs

    def index_from_sentence(self, sentence: str) -> List[int]:
        return [self.word2index[word] for word in sentence.split()] + \
               [Vocabulary.EOS_token]

    def input_var(self, seq: List[str]) -> Tuple[torch.LongTensor, torch.Tensor]:
        indexes_batch: List[int] = [
            self.index_from_sentence(sentence) for sentence in seq
        ]
        lengths: torch.Tensor = torch.Tensor(
            [len(indexes) for indexes in indexes_batch]
        )
        pad_var: torch.LongTensor = torch.LongTensor(
            Vocabulary.zero_padding(indexes_batch)
        )
        return pad_var, lengths

    def output_var(self, seq: List[str]
                   ) -> Tuple[torch.LongTensor, torch.BoolTensor, int]:
        indexes_batch: List[int] = [
            self.index_from_sentence(sentence) for sentence in seq
        ]
        max_target_len: int = max(
            [len(indexes) for indexes in indexes_batch]
        )
        pad_list: List[List[str]] = Vocabulary.zero_padding(indexes_batch)
        mask: torch.BoolTensor = torch.BoolTensor(
            Vocabulary.binary_matrix(pad_list)
        )
        return torch.LongTensor(pad_list), mask, max_target_len

    def batch_to_train_data(self, pair_batch: List[Pairs]):
        pair_batch = sorted(pair_batch, key=lambda x: len(x[0].split()), reverse=True)
        input_batch, output_batch = zip(*pair_batch)
        inp, inp_lengths = self.input_var(input_batch)
        out, mask, max_target_len = self.output_var(output_batch)
        return inp, inp_lengths, out, mask, max_target_len

    @staticmethod
    def zero_padding(seq: List[List[int]]) -> List[List[int]]:
        return list(zip_longest(*seq, fillvalue=Vocabulary.PAD_token))

    @staticmethod
    def binary_matrix(seq: List[List[int]]) -> List[List[str]]:
        return [[int(token != Vocabulary.PAD_token) for token in s]
                for s in seq]

    @staticmethod
    def filter_pair(pair: List[str], *, max_length: int = 10) -> bool:
        return all(len(sentence.split()) < max_length
                   for sentence in pair)
