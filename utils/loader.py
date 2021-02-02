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
from typing import List

import numpy as np

import utils.preprocessing as preprocessing
from utils.vocabulary import Pairs, Vocabulary


class Loader:
    """
    Class which downloads and reorganize
    corpus to format which can loaded into a
    vocabulary class.

    Attributes
    ----------
    config: dict
        Contains configuration of a class.
    data_dir: str
        Directory where corpus will be stored.
    movie_lines_fields: List[str]
        List of fields used in a movie lines file.
    movie_conversation_fields: List[str]
        List of fields used in a move converstions file.
    """
    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config: dict
            Dict with a configuration of a class.
        """
        self.config: dict = config
        self.data_dir: str = config["data"]["data_dir"]
        self.movie_lines_fields: List[str] =\
            ["lineID", "characterID", "movieID", "character", "text"]
        self.movie_conversation_fields: List[str] =\
            ["character1ID", "character2ID", "movieID", "utteranceIDs"]

        if not os.path.exists(os.path.join(self.data_dir, self.config["data"]["archive_name"])):
            self.download_corpus()

    def download_corpus(self) -> None:
        """
        Download compressed corpus and extracts it.
        """
        archive: str = os.path.join(self.data_dir, self.config["data"]["archive_name"])

        os.makedirs(self.data_dir, exist_ok=True)
        urllib.request.urlretrieve(self.config["data"]["url"], archive)

        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(self.data_dir)

        extracted_dir: str = os.path.join(self.data_dir, self.config["data"]["corpus_name"])
        for file in glob.glob(os.path.join(extracted_dir, "*")):
            shutil.move(file, self.data_dir)
        shutil.rmtree(extracted_dir)

    def format_movie_lines(self) -> str:
        """
        Format downloaded corpus in order
        to easily load it into a vocabulary object.

        Returns
        -------
        str
            Name of created file.
        """
        data_file: str = os.path.join(self.data_dir, self.config["data"]["formated_file"])
        if not os.path.exists(data_file):
            delimiter: str = str(codecs.decode("\t", "unicode_escape"))

            lines: dict = self.split_movie_lines(
                os.path.join(self.data_dir, self.config["data"]["lines_file"]))
            conversations: List[dict] = self.group_into_conversations(
                os.path.join(self.data_dir, self.config["data"]["conversations_file"]),
                lines)

            with open(data_file, "w", encoding="utf-8") as f:
                writer: csv.writer = csv.writer(f, delimiter=delimiter, lineterminator="\n")
                for pair in Loader.extract_sentence_pairs(conversations):
                    writer.writerow(pair)

        return data_file

    def load_prepared_data(self) -> Vocabulary:
        """
        Load formated corpus into a vocabulary object.

        Returns
        -------
        Vocabulary
            Vocabulary initialized with formated corpus.
        """
        with open(os.path.join(self.data_dir, self.config["data"]["formated_file"]),
                  encoding="utf-8") as f:
            lines: List[str] = f.read().strip().split('\n')

        pairs: Pairs = [
            [preprocessing.basic_preprocessing(s) for s in line.split('\t')]
            for line in lines
        ]
        vocabulary: Vocabulary = Vocabulary(pairs, self.config)
        return vocabulary

    def split_movie_lines(self, file_name: str) -> dict:
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
                    for i, field in enumerate(self.movie_lines_fields)
                }
                lines[line_obj['lineID']] = line_obj
        return lines

    def group_into_conversations(self, file_name: str, lines: dict) -> List[dict]:
        """
        Group movie lines into a conversations.

        Parameters
        ----------
        file_name:
            Name of a movie conversations file.
        lines:
            Movie lines loaded into a dict.
        Returns
        -------
        List[dict]
            Movie lines grouped into a conversations.
        """
        conversations: List[dict] = []
        with open(file_name, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values: List[str] = line.split(" +++$+++ ")
                conv_obj: dict = {
                    field: values[i]
                    for i, field in enumerate(self.movie_conversation_fields)
                }
                utterance_id_pattern: re.Pattern = re.compile("L[0-9]+")
                line_ids: List[str] = utterance_id_pattern.findall(conv_obj["utteranceIDs"])
                conv_obj["lines"] = [lines[line_id] for line_id in line_ids]

                conversations.append(conv_obj)
        return conversations

    @staticmethod
    def extract_sentence_pairs(conversations: List[dict]) -> Pairs:
        """
        Construct list of pair of sentences from a conversations.

        Parameters
        ----------
            Conversations to be reorganized into `Pairs`.
        Returns
        -------
        Pairs
            List of sentences pairs.
        """
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
        """
        Return random lines from a file.

        Parameters
        ----------
        file_name: str
            Name of the file.
        Returns
        -------
        str
            Random lines joined into a str.
        """
        with open(file_name, "rb") as f:
            lines: List[str] = f.readlines()

        return "".join(f"{line}\n" for line in np.random.choice(lines, size=10))


if __name__ == "__main__":
    import yaml
    with open("../settings/config.yaml", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error during config load: {str(e)}")
    loader = Loader(config)
    # file = loader.format_movie_lines()

    voc = loader.load_prepared_data()
    small_batch_size: int = 5
    batches = voc.batch_to_train_data(
        [random.choice(voc.pairs) for _ in range(small_batch_size)])
    inp, inp_lengths, out, mask, max_target_len = batches
