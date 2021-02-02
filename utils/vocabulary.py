"""
File contains vocabulary class which contains mapping from words
to indexes and reverse mapping, count of each word and total number of
words.
"""
from itertools import zip_longest
from typing import Callable, List, Tuple

import torch

Pairs = List[List[str]]


class Vocabulary:
    """
    Vocabulary class containing word2index and reverse mapping.
    Also have method containing sentence transformation into torch
    tensors.

    Attributes
    ----------
    pairs: Pairs
        List of pairs of sentences representing input and output of a
        model.
    config: dict
        Dictionary with configuration of a vocabulary.
    num_words: int
        Number of words in a vocabulary.
    word2index: dict
        Mapping from a word to index
    word2count: dict
        Mapping from a word to a number of occurrences of this word in
        a vocabulary.
    index2word: dict
        Mapping form index to a word.
    """
    def __init__(self, pairs: Pairs, config: dict) -> None:
        """
        Parameters
        ----------
        pairs: Pairs
            List of pairs of sentences.
        config: dict
            Dict containing parameters of a vocabulary.
        """
        self.pairs: Pairs = pairs
        self.config: dict = config
        self.num_words: int = 3
        self.word2index: dict = {}
        self.word2count: dict = {}
        self.index2word: dict = {}

        self.reset()
        self.filter_all_pairs()
        self.add_pairs()
        self.trim_rare_words()

    def reset(self) -> None:
        """
        Reset vocabulary to initial state.
        """
        self.num_words = 3
        self.word2index = self.word2count = {}
        self.index2word = {
            self.config["vocabulary"]["pad_token"]: "PAD",
            self.config["vocabulary"]["sos_token"]: "SOS",
            self.config["vocabulary"]["eos_token"]: "EOS"
        }

    def add_pairs(self) -> None:
        """
        Iterates over pairs and add them to a mapping.
        """
        for pair in self.pairs:
            for sentence in pair:
                for word in sentence.split(' '):
                    self.add_word(word)

    def add_word(self, word: str) -> None:
        """
        Add given word to a existing mappings.
        Parameters
        ----------
        word: str
            Word given to a mapping.
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def filter_pair(self, pair: List[str]) -> bool:
        """
        Checks if length of sentences in given pair are
        smaller than `max_length`.

        Parameters
        ----------
        pair: List[str]
            Pair of sentences.
        Returns
        -------
        bool
            True if pair should be kept.
        """
        max_len: int = self.config["vocabulary"]["max_length"]
        return len(pair[0].split()) < max_len and len(pair[1].split()) < max_len

    def filter_all_pairs(self) -> None:
        """
        Filter all pairs if they are they are longer than given max length.
        """
        self.pairs = [pair for pair in self.pairs if self.filter_pair(pair)]

    def trim(self) -> None:
        """
        Trim words from mappings if they
        don't occur often enough.
        """
        keep_words: List[str] = [
            k for k, v in self.word2count.items()
            if v >= self.config["vocabulary"]["min_count_to_trim"]
        ]

        self.reset()
        for word in keep_words:
            self.add_word(word)

    def trim_rare_words(self) -> None:
        """
        Trim rare words from kept pairs and
        remove sentences form pairs if words don't
        occur in mapping
        """
        self.trim()

        keep_sentence: Callable[[str], bool] = \
            lambda sentence: all(word in self.word2index for word in sentence.split(' '))
        keep_pairs: Pairs = [
            pair for pair in self.pairs
            if keep_sentence(pair[0]) and keep_sentence(pair[1])
        ]
        self.pairs = keep_pairs

    def zero_padding(self, seq: List[List[int]]) -> List[List[int]]:
        """
        Zero pad all sentences to a sentence of longest length.

        Parameters
        ----------
        seq: List[List[int]]
            Sentences in a token format.
        Returns
        -------
        List[List[int]]
            Sentences after zero padding.
        """
        return list(zip_longest(*seq, fillvalue=self.config["vocabulary"]["pad_token"]))

    def binary_matrix(self, seq: List[List[int]]) -> List[List[str]]:
        """
        Return binary mask from a list of sequences in token format.
        True is set if token is different from `pad_token`.

        Parameters
        ----------
        seq: List[List[int]]
            List of sentences in a token format.
        Returns
        -------
            Binary mask for a given list, 0 if token equal to `pad_token`
            1 otherwise.
        """
        return [[int(token != self.config["vocabulary"]["pad_token"]) for token in s]
                for s in seq]

    def index_from_sentence(self, sentence: str) -> List[int]:
        """
        Converts sentence to a list of tokens.

        Parameters
        ----------
        sentence: str
            Sentence to be converted.
        Returns
        -------
        List[int]
            Sentence in a index list form.
        """
        return [self.word2index[word] for word in sentence.split()] + \
               [self.config["vocabulary"]["eos_token"]]

    def input_var(self, seq: List[str]) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Converts list of str to a format suitable for a torch encoder model.

        Parameters
        ----------
        seq: List[str]
             List of sentences to be converted into model accepted format.

        Returns
        -------
        Tuple[torch.LongTensor, torch.Tensor]
            Tuple of tensors where firs is a tensor of list of sentences
            in a index format and second is a length of those sentences.
        """
        indexes_batch: List[int] = [
            self.index_from_sentence(sentence) for sentence in seq
        ]
        lengths: torch.Tensor = torch.Tensor(
            [len(indexes) for indexes in indexes_batch]
        )
        pad_var: torch.LongTensor = torch.LongTensor(
            self.zero_padding(indexes_batch)
        )
        return pad_var, lengths

    def output_var(self, seq: List[str]
                   ) -> Tuple[torch.LongTensor, torch.BoolTensor, int]:
        """
        Converts list of str into a format suitable for decoder model.

        Parameters
        ----------
        seq: List[str]
            List of output sentences.
        Returns
        -------
        Tuple[torch.LongTensor, torch.BoolTensor, int]
            Tuple formed from tensor of sentences in token format,
            boolean mask of those sentences and their max length.
        """
        indexes_batch: List[int] = [
            self.index_from_sentence(sentence) for sentence in seq
        ]
        max_target_len: int = max(
            [len(indexes) for indexes in indexes_batch]
        )
        pad_list: List[List[str]] = self.zero_padding(indexes_batch)
        mask: torch.BoolTensor = torch.BoolTensor(
            self.binary_matrix(pad_list)
        )
        return torch.LongTensor(pad_list), mask, max_target_len

    def batch_to_train_data(self, pair_batch: List[Pairs]
                            ) -> Tuple[torch.LongTensor, torch.Tensor,
                                       torch.LongTensor, torch.BoolTensor, int]:
        """
        Given a batch of sentence's pairs transform them into a
        encoder and decoder format.

        Parameters
        ----------
        pair_batch: List[Pairs]
            Batch of sentence's pairs.
        Returns
        -------
        tuple
            Tuple of values suitable to be forwarded to a torch model.
        """
        pair_batch = sorted(pair_batch, key=lambda x: len(x[0].split()), reverse=True)
        input_batch, output_batch = zip(*pair_batch)
        inp, inp_lengths = self.input_var(input_batch)
        out, mask, max_target_len = self.output_var(output_batch)
        return inp, inp_lengths, out, mask, max_target_len
