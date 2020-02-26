import re
import string
from typing import List

import fastai.text.transform
from nltk.tokenize import TweetTokenizer
from nn_toolkit.tokenizer import SimpleTokenizer
import spacy


_WHITESPACE = set(string.whitespace)


def preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r"\u2026", '...', text)
    text = re.sub(r'l{2,}', 'll', text)
    text = re.sub(r'm{2,}', 'mm', text)
    text = re.sub(r's{2,}', 'ss', text)
    text = re.sub(r'e{2,}', 'ee', text)
    text = re.sub(r'o{2,}', 'oo', text)
    text = re.sub(r't{2,}', 'tt', text)
    text = re.sub(r'f{2,}', 'ff', text)
    text = re.sub(r'p{2,}', 'pp', text)
    text = re.sub(r'y{2,}', 'y', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\?{2,}', '??', text)
    text = re.sub(r'\${2,}', '$$', text)
    text = re.sub(r'\x90', ' ', text)
    text = re.sub(r'[\x80-\x9f]', '', text)
    text = re.sub(r'#', ' #', text)
    text = re.sub(r'@', ' @', text)
    return text


def postprocessing(tokens: List[str]) -> List[str]:
    tokens = [clean_token(t) for t in tokens]
    tokens = replace_nonascii(tokens)
    tokens = [t for t in tokens if t not in _WHITESPACE]
    return tokens


def replace_nonascii(tokens: List[str]) -> List[str]:
    """Replace long tokens that contain mostly nonascii characters.

    Tokens whose length is less than `min_allowable_length` will be unchanged.
    """
    min_allowable_length = 4

    def is_ascii(token):
        n_chars = len(token)
        pct_ascii = sum([ord(c)<128 for c in token]) / n_chars
        return pct_ascii > 0.9

    for i, token in enumerate(tokens):
        if not is_ascii(token) and len(token) > min_allowable_length:
            tokens[i] = '<nonascii>'
    return tokens


def unhashtag(tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        if token.startswith('#') and len(token) > 1:
            tokens[i] = token[1:]
    return tokens


def split_hashtags(tokens: List[str]) -> List[str]:
    new_tokens = []
    for token in tokens:
        if token.startswith('#'):
            new_tokens.append('#')
            if len(token) > 1:
                new_tokens.append(token[1:])
        else:
            new_tokens.append(token)
    return new_tokens


def replace_repetitions(tokens: List[str]) -> List[str]:
    """Replace repeating tokens."""
    if len(tokens) <= 1:
        return tokens
    new_tokens = []
    slow, fast = 0, 1
    reps = 0
    while fast < len(tokens):
        slow_token = tokens[slow]
        fast_token = tokens[fast]
        if fast_token == slow_token:
            reps += 1
            fast += 1
            if fast == len(tokens):
                new_tokens += ['<rep>']
        else:
            if reps > 0:
                new_tokens += ['<rep>', slow_token]
                slow = fast
                fast += 1
                reps = 0
            else:
                new_tokens += [slow_token]
                slow += 1
                fast += 1
    new_tokens.append(fast_token)
    return new_tokens


def clean_token(token: str) -> str:
    if token.startswith('//t.co'):
        return '<//t.co>'
    elif token.startswith('http'):
        return '<url>'
    elif re.search(r'[\U0001f600-\U0001f650]', token) is not None:
        return '<emoji>'
    return token


class TweetSpacyTokenizer(fastai.text.transform.SpacyTokenizer):
    def tokenizer(self, text: str) -> List[str]:
        tokens = []
        for token in self.tok.tokenizer(text):
            tokens.append(token.text)
            if token.whitespace_:
                tokens.append(token.whitespace_)
        return tokens


class ProjectTokenizer(fastai.text.transform.Tokenizer):
    def __init__(self) -> None:
        super().__init__(tok_func=TweetSpacyTokenizer, pre_rules=[fastai.text.transform.fix_html, preprocessing], post_rules=[postprocessing])
