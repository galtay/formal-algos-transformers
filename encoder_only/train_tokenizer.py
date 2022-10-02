from pathlib import Path
import datasets
import torch.nn as nn
from transformers import AutoTokenizer


class BatchIterator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i : i + self.batch_size]["paragraph"]


ref_model = 'bert-base-uncased'

ds = datasets.load_dataset(
    "gabrielaltay/hacdc-wikipedia", name="text-intros", split="every"
)

batch_size = 5_000
batch_iterator = BatchIterator(ds, batch_size)

vocab_sizes = [5_000, 10_000, 20_000, 40_000, 80_000]
vocab_tags = ["5k", "10k", "20k", "40k", "80k"]

min_frequency = 0
limit_alphabet = 512

for vocab_size, vocab_tag in zip(vocab_sizes, vocab_tags):

    print(ref_model, vocab_size, vocab_tag)

    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model)
    tokenizer = ref_tokenizer.train_new_from_iterator(
        batch_iterator,
        vocab_size,
        add_prefix_space = False
        min_frequency=min_frequency,
        limit_alphabet=limit_alphabet,
    )
    out_path = Path(f"ref-{ref_model}") / Path(f'tokenizer-hacdc-wikipedia-vocab-{vocab_tag}')
    tokenizer.save_pretrained(out_path)
