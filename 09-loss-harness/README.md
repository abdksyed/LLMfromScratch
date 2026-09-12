# Loss Harness

This [notebook](01_understanding_llm_loss.ipynb) walks through how next-token loss is computed for an LLM. We will be using an existing pre-trained model (`Qwen/Qwen3.5-0.8B`). The model's hidden dimension is 1024 with the size of the vocabulary being 248,320.

The basic training step is:

```text
out = model(tokens)
logits = out.logits
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
opt.zero_grad()
loss.backward()
opt.step()
```

The model at position `t` predicts the token at position `t+1`. We drop the last logit because there is no next token to compare it with. We also ignore padding and the boundary between two packed documents.

You can view the full executed notebook at [syedak.com/understanding_llm_loss](https://syedak.com/understanding_llm_loss)

## Part 1

### Tensor shapes

For the input `"The capital of France is"` (5 tokens):

- `input_ids`: `[1, 5]` — batch size, sequence length
- logits: `[1, 5, 248320]` — batch size, sequence length, vocab size
- hidden states: `[1, 5, 1024]` — batch size, sequence length, model dimension
- attention maps: `[1, 8, 5, 5]` — batch size, number of heads, query length, key length

This model only has 6 full attention layers, so we get 6 attention maps, not 24.

For training we cut the last logit and the first token:

- `logits[:, :-1]` has shape `[batch, T-1, vocab]`
- `tokens[:, 1:]` has shape `[batch, T-1]`
- after `reshape(-1, ...)` both become one long list of predictions and targets

### Checking the shift with token strings

Printing token ids makes it easy to miss an off-by-one error. Printing the actual strings is clearer.

```text
position | input token | actual next token | model prediction
       0 | The         | Ġcapital           | Ġfollowing
       1 | Ġcapital    | Ġof                | Ġof
       2 | Ġof         | ĠFrance            | Ġthe
       3 | ĠFrance     | Ġis                | Ġis
       4 | Ġis         | (none)             | ĠParis
```

Position 4 predicts `Paris`, which is the next word after the input. We do not have a ground-truth label for that position, so it is not used in the loss. The `Ġ` character means a leading space.

### Masking padding

The sentence `"The capital of India is New Delhi."` has 8 tokens. We pad it to length 16 with `<|endoftext|>` and set those pad labels to `-100`.

After the shift there are 15 positions. If we do not mask padding, all 15 of them go into the loss. If we do mask padding, only 7 of them do.

The mean loss is `2.656` in both cases. That confirms the pad positions were ignored and did not change the average.

### Packing two documents

We put two sentences in one sequence, separated by `<|im_end|>`:

```text
The capital of India is New Delhi.<|im_end|>Taj Mahal is NOT situated in New Delhi.<|im_end|>
```

At the end of the first document, the model is asked to predict `T`, the first token of `Taj`. That prediction has a loss of `27.875`, which is much higher than the other tokens.

| setup | tokens in the loss | mean loss |
| --- | ---: | ---: |
| padding masked, boundary kept | 20 | 4.0625 |
| padding and boundary both masked | 19 | 2.7969 |

The mean drops because that one hard boundary token is removed. The model still learns to predict the end-of-sequence token. It does not learn to predict the first word of the next document.

### Perplexity of an untrained model

Perplexity is `exp(mean cross-entropy)`. A model that picks tokens at random should have:

- loss around `ln(248320) = 12.422`
- perplexity around the vocab size, `248,320`

A freshly initialized copy of the model gave loss `12.592` and perplexity `294,233`. That is close to the vocab size, so the loss code is aligned correctly.

The pretrained model on the same text had loss `2.320` and perplexity `10.17`.

### Tied vs untied head

This model shares the input embedding matrix with the output head. Both are `248320 x 1024`, which is `254,279,680` parameters.

| setup | total parameters |
| --- | ---: |
| tied (what the model actually uses) | 852,985,920 |
| untied (separate output matrix) | 1,107,265,600 |
| extra cost of untying | 254,279,680 |

Keeping them tied saves about 254 million parameters.

### Chunked cross entropy

Normal cross entropy builds a full `[number of tokens, vocab size]` logit tensor. For 631 tokens that uses a lot of memory. A chunked version scores 32 tokens at a time and averages the same way.

| method | peak memory | time | loss |
| --- | ---: | ---: | ---: |
| ordinary | 1494.86 MB | 8.26 ms | 2.325397 |
| chunked (32 tokens) | 75.78 MB | 20.82 ms | 2.325397 |

The losses match. Chunking uses about 20 times less peak memory and takes about 2.5 times longer.

## Part 2

We add a second output head that predicts token `t+2` instead of `t+1`. It has the same shape as the original head. We copy the original weights, then keep the two heads independent.

Both heads are trained together for 100 steps on 22 short sentences.

| step | t+1 loss | t+2 loss | sum |
| ---: | ---: | ---: | ---: |
| 0 | 3.8985 | 9.2174 | 13.1159 |
| 50 | 0.7429 | 0.7333 | 1.4763 |
| 100 | 0.6516 | 0.6435 | 1.2951 |

At the start, the `t+2` loss is much higher (`9.22` vs `3.90`). That is expected. The copied weights are already good at predicting the next token, not the token after that.

During training both losses fall quickly. By step 100 they are almost the same. The dataset is tiny, so the model is mostly memorizing the 22 sentences. On a training sentence both heads look correct. On a new sentence they start predicting `the` over and over.
