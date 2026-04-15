import os as _os
import urllib.request as _urllib_request
import pyarrow.parquet as _pq
import torch as _torch

_URL_GLUE = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/{task}/{file}"


def _load_glue_parquet(task, filename, root):
    _os.makedirs(root, exist_ok=True)
    path = _os.path.join(root, f"glue_{task}_{filename}")
    if not _os.path.exists(path):
        print(f"Downloading GLUE {task} {filename}...")
        _urllib_request.urlretrieve(_URL_GLUE.format(task=task, file=filename), path)
    return _pq.read_table(path).to_pylist()


class CoLA(_torch.utils.data.Dataset):
    """Corpus of Linguistic Acceptability. Single sentence grammatical acceptability.

    Each item is a (sentence, label) tuple:
        sentence — input sentence
        label    — int (0=unacceptable, 1=acceptable)

    Metric: Matthews correlation coefficient.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("cola", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["sentence"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SST2(_torch.utils.data.Dataset):
    """Stanford Sentiment Treebank, binary. Single sentence sentiment classification.

    Each item is a (sentence, label) tuple:
        sentence — input sentence
        label    — int (0=negative, 1=positive)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("sst2", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["sentence"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MRPC(_torch.utils.data.Dataset):
    """Microsoft Research Paraphrase Corpus. Sentence pair paraphrase detection.

    Each item is a (sentence1, sentence2, label) tuple:
        sentence1 — first sentence
        sentence2 — second sentence
        label     — int (0=not paraphrase, 1=paraphrase)

    Metric: F1 / accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("mrpc", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["sentence1"], row["sentence2"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class STSB(_torch.utils.data.Dataset):
    """Semantic Textual Similarity Benchmark. Sentence pair similarity regression.

    Each item is a (sentence1, sentence2, label) tuple:
        sentence1 — first sentence
        sentence2 — second sentence
        label     — float (0.0 to 5.0)

    Metric: Pearson / Spearman correlation.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("stsb", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["sentence1"], row["sentence2"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QQP(_torch.utils.data.Dataset):
    """Quora Question Pairs. Duplicate question detection.

    Each item is a (question1, question2, label) tuple:
        question1 — first question
        question2 — second question
        label     — int (0=not duplicate, 1=duplicate)

    Metric: F1 / accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("qqp", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["question1"], row["question2"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MNLI(_torch.utils.data.Dataset):
    """Multi-Genre Natural Language Inference. Sentence pair 3-way classification.

    Each item is a (premise, hypothesis, label) tuple:
        premise    — premise sentence
        hypothesis — hypothesis sentence
        label      — int (0=entailment, 1=neutral, 2=contradiction)

    Metric: accuracy.
    """
    def __init__(self, split="validation_matched", root=".data"):
        assert split in {"train", "validation_matched", "validation_mismatched", "test_matched", "test_mismatched"}
        rows = _load_glue_parquet("mnli", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["premise"], row["hypothesis"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QNLI(_torch.utils.data.Dataset):
    """Question Natural Language Inference. Question-sentence entailment.

    Each item is a (question, sentence, label) tuple:
        question — question string
        sentence — answer sentence
        label    — int (0=entailment, 1=not_entailment)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("qnli", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["question"], row["sentence"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RTE(_torch.utils.data.Dataset):
    """Recognizing Textual Entailment. Sentence pair binary entailment.

    Each item is a (sentence1, sentence2, label) tuple:
        sentence1 — premise
        sentence2 — hypothesis
        label     — int (0=entailment, 1=not_entailment)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_glue_parquet("rte", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["sentence1"], row["sentence2"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
