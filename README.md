# infinite_molière

This toy project is a mere application of the excellent tutorial [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy. Here, I train a transformer-based decoder-only model to generate Molière-like text, using the (almost) complete corpus on [Molière](https://en.wikipedia.org/wiki/Moli%C3%A8re)'s work.

This model uses BERT's WordPiece tokenizer, which is sub-optimal for this (french) corpus. Adopting a better-suited tokenizer is a proximal improvement to this project.