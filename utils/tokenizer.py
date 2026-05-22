from typing import Dict, List, Tuple

import tiktoken

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer



class OpenAITokenizerWrapper(BaseTokenizer):

    model_input_names = ["input_ids"]

    def __init__(
        self,
        model_name: str = "cl100k_base",
        max_length: int = 300,
        **kwargs,
    ):

        super().__init__(
            model_max_length=max_length,
            **kwargs,
        )

        self.encoding = tiktoken.get_encoding(model_name)

        self._vocab_size = 100277

    # -----------------------------------
    # REQUIRED METHODS
    # -----------------------------------

    def tokenize(self, text: str, **kwargs) -> List[str]:

        return [
            str(token)
            for token in self.encoding.encode(text)
        ]

    def _tokenize(self, text: str) -> List[str]:

        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:

        return int(token)

    def _convert_id_to_token(self, index: int) -> str:

        return str(index)

    def get_vocab(self) -> Dict[str, int]:

        return {
            str(i): i
            for i in range(self.vocab_size)
        }

    @property
    def vocab_size(self) -> int:

        return self._vocab_size

    def save_vocabulary(self, *args, **kwargs) -> Tuple[str]:

        return ()

    # -----------------------------------
    # TOKEN COUNTING
    # -----------------------------------

    def encode(
        self,
        text,
        **kwargs,
    ):

        return self.encoding.encode(text)

    def decode(
        self,
        token_ids,
        **kwargs,
    ):

        return self.encoding.decode(token_ids)

    # -----------------------------------
    # HF STYLE LOADER
    # -----------------------------------

    @classmethod
    def from_pretrained(cls, *args, **kwargs):

        return cls(**kwargs)