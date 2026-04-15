import os as _os
import urllib.request as _urllib_request
import pyarrow.parquet as _pq
import torch as _torch

_URL_SUPERGLUE = "https://huggingface.co/datasets/aps/super_glue/resolve/main/{task}/{file}"


def _load_superglue_parquet(task, filename, root):
    _os.makedirs(root, exist_ok=True)
    path = _os.path.join(root, f"superglue_{task}_{filename}")
    if not _os.path.exists(path):
        print(f"Downloading SuperGLUE {task} {filename}...")
        _urllib_request.urlretrieve(_URL_SUPERGLUE.format(task=task, file=filename), path)
    return _pq.read_table(path).to_pylist()


class BoolQ(_torch.utils.data.Dataset):
    """Boolean Questions. Passage + yes/no question binary classification.

    Each item is a (passage, question, label) tuple:
        passage  — context paragraph
        question — yes/no question
        label    — int (0=false, 1=true)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("boolq", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["passage"], row["question"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class CB(_torch.utils.data.Dataset):
    """CommitmentBank. Premise-hypothesis 3-way NLI.

    Each item is a (premise, hypothesis, label) tuple:
        premise    — premise paragraph
        hypothesis — hypothesis statement
        label      — int (0=entailment, 1=contradiction, 2=neutral)

    Metric: accuracy / macro-F1.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("cb", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["premise"], row["hypothesis"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class COPA(_torch.utils.data.Dataset):
    """Choice of Plausible Alternatives. Causal reasoning 2-way selection.

    Each item is a (premise, choice1, choice2, question, label) tuple:
        premise  — premise sentence
        choice1  — first candidate
        choice2  — second candidate
        question — "cause" or "effect"
        label    — int (0=choice1, 1=choice2)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("copa", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["premise"], row["choice1"], row["choice2"], row["question"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MultiRC(_torch.utils.data.Dataset):
    """Multi-Sentence Reading Comprehension. Multi-answer QA.

    Each item is a (paragraph, question, answer, label, question_id) tuple:
        paragraph   — context paragraph
        question    — question string
        answer      — candidate answer
        label       — int (0=incorrect, 1=correct)
        question_id — (paragraph_idx, question_idx) grouping answers per question

    Metric: F1_a (pooled F1) / EM (per-question exact match).
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("multirc", f"{split}-00000-of-00001.parquet", root)
        self.examples = []
        for row in rows:
            idx_info = row["idx"]
            question_id = (idx_info["paragraph"], idx_info["question"])
            self.examples.append((row["paragraph"], row["question"], row["answer"], row["label"], question_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ReCoRD(_torch.utils.data.Dataset):
    """Reading Comprehension with Commonsense Reasoning. Cloze-style entity QA.

    Each item is a (passage, query, entities, answers) tuple:
        passage  — context paragraph
        query    — cloze query containing @placeholder
        entities — list of candidate entity strings
        answers  — list of gold answer strings (subset of entities)

    Metric: token-level F1 / EM (max over answer mentions).
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("record", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["passage"], row["query"], row["entities"], row["answers"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RTE(_torch.utils.data.Dataset):
    """Recognizing Textual Entailment. Sentence pair binary entailment.

    Each item is a (premise, hypothesis, label) tuple:
        premise    — premise sentence
        hypothesis — hypothesis sentence
        label      — int (0=entailment, 1=not_entailment)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("rte", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["premise"], row["hypothesis"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class WiC(_torch.utils.data.Dataset):
    """Word in Context. Word-sense disambiguation across two sentences.

    Each item is a (word, sentence1, sentence2, label) tuple:
        word      — target word
        sentence1 — first sentence containing the word
        sentence2 — second sentence containing the word
        label     — int (0=different sense, 1=same sense)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("wic", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["word"], row["sentence1"], row["sentence2"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class WSC(_torch.utils.data.Dataset):
    """Winograd Schema Challenge. Pronoun coreference binary classification.

    Each item is a (text, span1_text, span2_text, label) tuple:
        text       — full sentence
        span1_text — candidate antecedent
        span2_text — pronoun
        label      — int (0=not coreferent, 1=coreferent)

    Metric: accuracy.
    """
    def __init__(self, split="validation", root=".data"):
        assert split in {"train", "validation", "test"}
        rows = _load_superglue_parquet("wsc", f"{split}-00000-of-00001.parquet", root)
        self.examples = [(row["text"], row["span1_text"], row["span2_text"], row["label"]) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
