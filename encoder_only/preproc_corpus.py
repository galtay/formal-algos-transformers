from itertools import chain
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

l_max = 128
num_proc = 24
map_batch_size = 1_000
ref_model = "bert_base_uncased"


vocab_tags = ["5k", "10k", "20k", "40k", "80k"]
ds = datasets.load_dataset(
    "gabrielaltay/hacdc-wikipedia", name="text", split="every"
)


for vocab_tag in vocab_tags:

    print(ref_model, vocab_tag)

    tokenizer_checkpoint = Path(f"ref-{ref_model}") / Path(f'tokenizer-hacdc-wikipedia-vocab-{vocab_tag}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    out_path = Path(f"ref-{ref_model}") / Path(
        f"preproc-hacdc-wikipedia-vocab-{vocab_tag}"
    )

    def tokenize_function(examples):
        return tokenizer(examples["paragraph"])

    ds_tokenized = ds.map(
        tokenize_function,
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
        remove_columns=["paragraph"],
    )
    print("SAVING TOKENIZED TO DISK")
    ds_tokenized.save_to_disk(out_path / "tokenized")

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the
        # model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= l_max:
            total_length = (total_length // l_max) * l_max
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + l_max] for i in range(0, total_length, l_max)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    ds_lm = ds_tokenized.map(
        group_texts, batched=True, batch_size=map_batch_size, num_proc=num_proc,
    )

    print("SAVING LM TO DISK")
    ds_lm.save_to_disk(out_path / "lm")
