from pathlib import Path

import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import wandb

from formal_algos_transformers.fat_multi_head_attention import MultiHeadAttention
from formal_algos_transformers.embeddings import ContentEmbeddings
from formal_algos_transformers.embeddings import PositionEncodings
from formal_algos_transformers.bert_like import (
    Embeddings,
    EncoderBlock,
    EncoderStack,
    EncoderHeadless,
    EncoderMlmHead,
    EncoderForMlm,
    PointwiseFeedForward,
)
from formal_algos_transformers.bert_like import get_pad_mask, make_bert_like_encoder


wandb.init(project="transformers-hacdc")


l_max = 128
num_proc = 24
map_batch_size = 10_000
mlm_probability = 0.15
ref_model = "bert-base-uncased"
vocab_tag = "5k"

train_batch_size = 128
test_batch_size = 512

tokenizer_checkpoint = f"ref-{ref_model}/tokenizer-hacdc-wikipedia-vocab-{vocab_tag}"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

ds_checkpoint = (
    Path(f"ref-{ref_model}") / Path(f"preproc-hacdc-wikipedia-vocab-{vocab_tag}") / "lm"
)
ds_lm = datasets.load_from_disk(ds_checkpoint)
ds_lm_shuffled = ds_lm.shuffle(seed=42)
dsd_lm = ds_lm_shuffled.train_test_split(test_size=0.01)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=mlm_probability
)

dl_train = DataLoader(
    dsd_lm["train"].with_format("torch"),
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=data_collator,
)
dl_test = DataLoader(
    dsd_lm["test"].with_format("torch"),
    batch_size=test_batch_size,
    collate_fn=data_collator,
)


embd_size = 512
n_layers = 4
n_h = 8
prenorm = True
bias = True
dropout_proba = 0.1

model = make_bert_like_encoder(
    tokenizer.vocab_size,
    l_max,
    embd_size=embd_size,
    n_layers=n_layers,
    n_h=n_h,
    prenorm=prenorm,
    bias=bias,
    dropout_proba=dropout_proba,
)


# prevent weight decay on bias, embeddings, and layernorms
no_decay = [
    "bias",
    "content.embedding.weight",
    "norm.weight",
    "norm_mha.weight",
    "norm_ff.weight",
]

no_decay_parameters = []
decay_parameters = []
for name, params in model.named_parameters():
    print("name: ", name)
    if not any(nd in name for nd in no_decay):
        decay_parameters.append(params)
    else:
        print(" *** no decay name: ", name)
        no_decay_parameters.append(params)

learning_rate = 5.0e-5
weight_decay = 0.0
optimizer_grouped_parameters = [
    {"params": decay_parameters, "weight_decay": weight_decay,},
    {"params": no_decay_parameters, "weight_decay": 0.0,},
]


wandb.config = {
    "n_v": tokenizer.vocab_size,
    "l_max": l_max,
    "embd_size": embd_size,
    "n_layers": n_layers,
    "n_h": n_h,
    "prenorm": prenorm,
    "bias": bias,
    "dropout_proba": dropout_proba,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
}

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fct = CrossEntropyLoss()

device = torch.device("cuda:1")
model = model.to(device)
num_train_steps_report = 50
num_train_steps_eval = 1_000
num_train_steps_checkpoint = 5_000

num_samples_seen = 0
num_batches_seen = 0

wandb.log(
    {
        "content_embedding_weights": wandb.Histogram(
            model.encoder_headless.embeddings.content.embedding.weight.cpu()
            .detach()
            .numpy()
        )
    }
)



class RunningQuantities:

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = 0
        self.batches = 0
        self.steps = 0
        self.loss = 0


rq = RunningQuantities()

tot_samples = 0
tot_batches = 0
tot_steps = 0

model_checkpoint_base = (
    Path(f"ref-{ref_model}") / Path(f"model-hacdc-wikipedia-vocab-{vocab_tag}")
)

for epoch in range(1):
    model.train()

    for step, batch in enumerate(dl_train):

        x_mask = get_pad_mask(
            batch["attention_mask"],
            batch["attention_mask"],
        ).to(device)
        x = batch["input_ids"].to(device)
        logits = model(x, x_mask)
        labels = batch["labels"].to(device)

        # combine the batch and sequence dimension
        loss = loss_fct(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        num_samples_in_batch = x.shape[0]
        rq.samples += num_samples_in_batch
        rq.batches += 1
        rq.steps += 1
        rq.loss += loss.item() * num_samples_in_batch

        tot_samples += num_samples_in_batch
        tot_batches += 1
        tot_steps += 1

        if (
            step % num_train_steps_report == 0 and step != 0
        ):
            print(
                f"[{epoch + 1}, {step:5d}] rolling loss/sample: {rq.loss / rq.samples:.3f}"
            )
            wandb.log(
                {
                    "train_loss": rq.loss / rq.samples,
                    "sample": tot_samples,
                    "batch": tot_batches,
                    "epoch": epoch,
                }
            )
            rq.reset()


        if (
            step % num_train_steps_eval == 0 and step != 0
        ):  # eval every num_train_steps_eval mini-batches

            model.eval()
            eval_losses = []
            for step, batch in enumerate(dl_test):
                with torch.no_grad():
                    x_mask = get_pad_mask(
                        batch["attention_mask"],
                        batch["attention_mask"]
                    ).to(device)
                    x = batch["input_ids"].to(device)
                    logits = model(x, x_mask)
                    labels = batch["labels"].to(device)

                    # combine the batch and sequence dimension
                    loss = loss_fct(
                        logits.view(-1, tokenizer.vocab_size), labels.view(-1)
                    )
                    eval_losses.append(loss.item())

            eval_loss = sum(eval_losses) / len(eval_losses)
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "sample": tot_samples,
                    "batch": tot_batches,
                    "epoch": epoch,
                }
            )
            model.train()
