# nanotss: a byte-level language model in one C file, with test-time training

`nanotss_blt.c` is the whole thing: model, trainer, test-time training (TTT), sampling and tools. Plain C with OpenMP,
CPU only, no dependencies. `model.bin` is the best checkpoint so far.

## The model

- **Architecture.** Byte-level, 1.19M parameters. The layers are SSM tree, gated FFF tree, SSM tree, gated FFF tree.
  - The SSM trees are routed binary trees whose nodes each hold a selective-decay state.
  - Keys and queries are computed per tree, with QK-norm and rotary.
  - Weights are ternary and activations int8.
- **Training.** Blockwise Muon, 20 minutes on 8 vCPU, on TinyShakespeare (the first 90%). The log is in
  `model_train.log`.
- **Scores (bits per byte) on TinyShakespeare validation.**

  | setting | BPC |
  |---|---|
  | no TTT | 2.206 |
  | online TTT, 2 steps | 2.054 |
  | session default (TTT 1 step + match model) | 2.036 |

- **Unseen plays** (`texts/`, public domain), reading the first half and scoring the second half:

  | setting | BPC |
  |---|---|
  | no TTT | 2.967 |
  | TTT | 2.259 |
  | TTT + match model | 2.154 |

The checkpoint stores its own configuration, so `--load model.bin` needs no architecture flags.

## Build

```sh
cc -O3 -march=native -fopenmp nanotss_blt.c -lm -o nanotss
```

## Use

```sh
# sample
./nanotss --load model.bin --prompt "ROMEO:" --sample 400 --temperature 0.8 --threads 4

# online TTT over a document: each 192-byte chunk is scored, then the weights take a few steps on it
./nanotss --load model.bin --ttt-eval texts/faustus.txt --ttt-offset 0 --ttt-lr 2e-4 --ttt-steps 2 --threads 4

# read the first half of some texts for 8 epochs, then score the rest
./nanotss --load model.bin --ttt-fit texts/faustus.txt,texts/tamburlaine2.txt --ttt-frac 0.5 --ttt-epochs 8 --ttt-lr 3e-4 --match 1

# interactive session on stdin
#   plain lines are context, learned online by TTT and added to the match model
#   commands: @@gen N [temp] [cands], @@file PATH, @@keep 0|1, @@stats, @@reset
./nanotss --load model.bin --session 1 --session-speed balanced --threads 4

# generate with the context as scratchpad: TTT on it, then retrieval-guided best-of-K rounds
./nanotss --load model.bin --refine 1 --refine-context texts/faustus.txt --prompt $'\n\nFAUSTUS:\n' \
  --refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 3 --refine-cands 8 --refine-keep 2 \
  --refine-retr 1 --refine-steps 8 --refine-temp 0.7 --refine-final-temp 0.7

# train your own (byte mode, Muon, QK-norm and rotary are the defaults)
./nanotss --data input.txt --seconds 1200 --threads 8 --save my.bin
```

`./nanotss --help` lists all options:
- `--ttt-*` (lr, steps, chunk, replay, retrieval, adaptive);
- `--match` (n-gram match model);
- `--mtp`, `--pause`, `--latent`;
- `--mode blt` (the byte-latent patcher stack);
- `--optimizer adamw|muon`.

Status lines are JSON on stderr.

## Findings in brief

- Long context helps through TTT, not through the recurrent state.
  - TTT gains about 0.2 BPC for each 4× more context, and is still improving at 100 KB.
  - With TTT, the 1.4M-parameter model beats a 4.2M one without it.
- One TTT epoch costs about 3 s per 100 KB on 4 threads.

## Sources

- TinyShakespeare is Karpathy's char-rnn dataset.
- The three plays in `texts/` are from Project Gutenberg, normalized to `NAME:` speaker lines.
