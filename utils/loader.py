"""
File containing code which download and extract in given
order text data.
"""
import re
import os
import csv
import glob
import shutil
import codecs
import zipfile
import random
import urllib.request
from typing import List, Tuple

import numpy as np

import utils.preprocessing as preprocessing
from utils.vocabulary import Pairs, Vocabulary


class Loader:
    url: str = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    data_name: str = "cornell_movie_dialogs_corpus.zip"
    corpus_name: str = "cornell movie-dialogs corpus"
    lines_file: str = "movie_lines.txt"
    conversations_file: str = "movie_conversations.txt"
    formated_file: str = "formatted_movie_lines.txt"
    movie_lines_fields: List[str] = ["lineID", "characterID", "movieID", "character", "text"]
    movie_conversation_fields: List[str] = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    def __init__(self, *,
                 data_dir: str = "data",
                 save_dir: str = "save") -> None:
        self.data_dir: str = data_dir
        self.save_dir: str = save_dir

        if not os.path.exists(os.path.join(data_dir, Loader.data_name)):
            self.download_corpus()

    def download_corpus(self) -> None:
        """
        Download compressed corpus and extracts it.
        """
        archive: str = os.path.join(self.data_dir, Loader.data_name)

        os.makedirs(self.data_dir, exist_ok=True)
        urllib.request.urlretrieve(Loader.url, archive)

        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(self.data_dir)

        # check if extracted dir ok
        extracted_dir: str = os.path.join(self.data_dir, Loader.corpus_name)
        for file in glob.glob(os.path.join(extracted_dir, "*")):
            shutil.move(file, self.data_dir)
        shutil.rmtree(extracted_dir)

    def format_movie_lines(self) -> str:
        data_file: str = os.path.join(self.data_dir, Loader.formated_file)
        delimiter: str = str(codecs.decode("\t", "unicode_escape"))

        lines: dict = Loader.split_movie_lines(
            os.path.join(self.data_dir, Loader.lines_file))
        conversations: List[dict] = Loader.group_into_conversations(
            os.path.join(self.data_dir, Loader.conversations_file),
            lines)

        with open(data_file, "w", encoding="utf-8") as f:
            writer: csv.writer = csv.writer(f, delimiter=delimiter, lineterminator="\n")
            for pair in Loader.extract_sentence_pairs(conversations):
                writer.writerow(pair)

        return data_file

    def load_prepared_data(self) -> Vocabulary:
        with open(os.path.join(self.data_dir, Loader.formated_file),
                  encoding="utf-8") as f:
            lines: List[str] = f.read().strip().split('\n')

        pairs: Pairs = [
            [preprocessing.basic_preprocessing(s) for s in line.split('\t')]
            for line in lines
        ]
        vocabulary: Vocabulary = Vocabulary(Loader.corpus_name, pairs)
        return vocabulary

    @staticmethod
    def split_movie_lines(file_name: str) -> dict:
        """
        Splits lines into a dictionary of fields.

        Parameters
        ----------
        file_name: str
            Name of the file that will be split.
        Returns
        -------
        dict
            Dict containing split file.
        """
        lines: dict = {}
        with open(file_name, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values: List[str] = line.split(" +++$+++ ")
                line_obj: dict = {
                    field: values[i]
                    for i, field in enumerate(Loader.movie_lines_fields)
                }
                lines[line_obj['lineID']] = line_obj
        return lines

    @staticmethod
    def group_into_conversations(file_name: str, lines: dict) -> List[dict]:
        conversations: List[dict] = []
        with open(file_name, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values: List[str] = line.split(" +++$+++ ")
                conv_obj: dict = {
                    field: values[i]
                    for i, field in enumerate(Loader.movie_conversation_fields)
                }
                utterance_id_pattern: re.Pattern = re.compile("L[0-9]+")
                line_ids: List[str] = utterance_id_pattern.findall(conv_obj["utteranceIDs"])
                conv_obj["lines"] = [lines[line_id] for line_id in line_ids]

                conversations.append(conv_obj)
        return conversations

    @staticmethod
    def extract_sentence_pairs(conversations: List[dict]) -> Pairs:
        pairs: Pairs = []
        for conversation in conversations:
            for inputs, targets in zip(conversation["lines"], conversation["lines"][1:]):
                input_line: str = inputs["text"].strip()
                target_line: str = targets["text"].strip()
                if input_line and target_line:
                    pairs.append([input_line, target_line])
        return pairs

    @staticmethod
    def get_random_lines(file_name: str) -> str:
        with open(file_name, "rb") as f:
            lines: List[str] = f.readlines()

        return "".join(f"{line}\n" for line in np.random.choice(lines, size=10))


if __name__ == "__main__":
    loader = Loader()
    file = loader.format_movie_lines()
    print(Loader.get_random_lines(file))

    print('============================')

    voc = loader.load_prepared_data()
    small_batch_size: int = 5
    batches = voc.batch_to_train_data(
        [random.choice(voc.pairs) for _ in range(small_batch_size)])
    inp, inp_lengths, out, mask, max_target_len = batches
    print("Input Variable", inp)
    print("lenghts", inp_lengths)
    print("target_varaible", out)
    print("mask", mask)
    print("max_target_len:", max_target_len)
