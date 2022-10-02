import torch
from transformers import AutoTokenizer
from formal_algos_transformers.bert_like import get_pad_mask, make_bert_like_encoder


l_max = 128
ref_model = "bert-base-uncased"
vocab_tag = "5k"

embd_size = 512
n_layers = 4
n_h = 8
prenorm = True
bias = True
dropout_proba = 0.1

tokenizer_checkpoint = f"ref-{ref_model}/tokenizer-hacdc-wikipedia-vocab-{vocab_tag}"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

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


model.load_state_dict(torch.load('sandy-tree-15/model-checkpoint-6796758.pt'))
model.eval()

device = torch.device("cpu")
text = "The bulding was [MASK] in January."
text = "The car [MASK] down the street."
tokenized = tokenizer(text, return_tensors="pt")
mask = get_pad_mask(tokenized['attention_mask'], tokenized['attention_mask']).to(device)
x = tokenized['input_ids'].to(device)
with torch.no_grad():
    logits = model(x, mask)

mask_pos = (tokenized['input_ids'][0] == tokenizer.mask_token_id).type(torch.int).argmax()
mask_logits = logits[0, mask_pos, :]
mask_probas = torch.softmax(mask_logits, dim=0)
sindxs = torch.argsort(-mask_probas)

print(text)
for ii in range(10):
    indx = sindxs[ii]
    print(ii, indx.item(), tokenizer.convert_ids_to_tokens(indx.item()), mask_probas[indx].item())
