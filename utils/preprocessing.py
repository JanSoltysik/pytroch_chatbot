"""
File containing text preprocessing required before
training a model.
"""
import re
import unicodedata


def unicode_to_ascii(text: str) -> str:
    """
    Converts unicode characters to plain ascii.

    Parameters
    ----------
    text: str
        Text to be converted.
    Returns
    -------
    str
        Text converted to plain ascii.
    """
    return ''.join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != 'Mn'
    )


def basic_preprocessing(text: str) -> str:
    """
    Performs basic text preprocessing:
    Parameters
    ----------
    text: str
        Text to be preprocessed.
    Returns
    -------
    str
        Preprocessed text.
    """
    preprocessed_text: str = unicode_to_ascii(text.lower().strip())
    preprocessed_text = re.sub(r"([.!?])", r" \1", preprocessed_text)
    preprocessed_text = re.sub(r"[^a-zA-Z.!?]+", r" ", preprocessed_text)
    preprocessed_text = re.sub(r"\s+", r" ", preprocessed_text).strip()
    return preprocessed_text

