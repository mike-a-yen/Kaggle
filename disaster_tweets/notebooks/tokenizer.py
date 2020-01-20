from typing import List

from nltk.tokenize import TweetTokenizer
from nn_toolkit.tokenizer import SimpleTokenizer


class ProjectTokenizer(SimpleTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = TweetTokenizer(strip_handles=False).tokenize

    def postprocessing(self, tokens: List[str]) -> List[str]:
        tokens = [self.clean_token(t) for t in tokens]
        return tokens

    def clean_token(self, token: str) -> str:
        if token.startswith('//t.co'):
            return '//t.co'
        elif token.startswith('@'):
            return '<@user-handle>'
        return token
