from typing import List

from nn_toolkit.tokenizer import SimpleTokenizer


class TweetTokenizer(SimpleTokenizer):
    def postprocessing(self, tokens: List[str]) -> List[str]:
        tokens = [self.clean_token(t) for t in tokens]
        return tokens

    def clean_token(self, token: str) -> str:
        if token.startswith('//t.co'):
            return '//t.co'
        return token
