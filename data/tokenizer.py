"""Character-level tokenizers compatible with a subset of the HF AutoTokenizer interface.

Two variants are provided, both fitting within 256 ids so they can be stored as uint8:
    "char-cased"  — printable ASCII plus whitespace, case preserved (vocab 104)
    "char-lower"  — same but uppercase A-Z removed; input is lowercased (vocab 78)

Use ``CharTokenizer.from_pretrained(name)`` or ``load_tokenizer(name)`` as a drop-in
replacement for ``AutoTokenizer.from_pretrained(name)``.
"""
import torch as _torch


_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

_CASED_NAMES = {"char", "char-cased", "char_cased", "char-case"}
_LOWER_NAMES = {"char-lower", "char_lower", "char-lowercase", "char_lowercase", "char-uncased"}


class CharTokenizer:
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        chars = ["\t", "\n", " "]
        for code in range(33, 127):
            ch = chr(code)
            if lowercase and "A" <= ch <= "Z":
                continue
            chars.append(ch)

        self._id_to_tok = list(_SPECIAL_TOKENS) + chars
        self._tok_to_id = {t: i for i, t in enumerate(self._id_to_tok)}
        assert len(self._id_to_tok) <= 256, f"vocab overflow: {len(self._id_to_tok)}"

        self.pad_token_id = self._tok_to_id[self.pad_token]
        self.unk_token_id = self._tok_to_id[self.unk_token]
        self.cls_token_id = self._tok_to_id[self.cls_token]
        self.sep_token_id = self._tok_to_id[self.sep_token]
        self.mask_token_id = self._tok_to_id[self.mask_token]
        self.bos_token_id = self._tok_to_id[self.bos_token]
        self.eos_token_id = self._tok_to_id[self.eos_token]
        self._special_ids = set(range(len(_SPECIAL_TOKENS)))

    @property
    def vocab_size(self):
        return len(self._id_to_tok)

    def __len__(self):
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        key = name.lower() if isinstance(name, str) else ""
        if key in _CASED_NAMES:
            return cls(lowercase=False)
        if key in _LOWER_NAMES:
            return cls(lowercase=True)
        raise ValueError(f"unknown char tokenizer: {name!r}")

    def encode(self, text, max_length=None, truncation=False, **kwargs):
        if self.lowercase:
            text = text.lower()
        unk = self.unk_token_id
        ids = [self._tok_to_id.get(ch, unk) for ch in text]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, _torch.Tensor):
            ids = ids.tolist()
        pieces = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in self._special_ids:
                continue
            if 0 <= i < len(self._id_to_tok):
                pieces.append(self._id_to_tok[i])
        return "".join(pieces)

    def add_special_tokens(self, mapping):
        # Our specials are fixed at construction; HF path only calls this when
        # mask_token_id is None, which never happens for us.
        assert False, "add_special_tokens should not be called on CharTokenizer"


def is_char_tokenizer_name(name):
    if not isinstance(name, str):
        return False
    key = name.lower()
    return key in _CASED_NAMES or key in _LOWER_NAMES


def load_tokenizer(name):
    if is_char_tokenizer_name(name):
        return CharTokenizer.from_pretrained(name)
    import transformers as _transformers
    return _transformers.AutoTokenizer.from_pretrained(name)
