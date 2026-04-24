| Model | rhythm div | rep ratio | pitch-hist dist | extras |
|-------|-----------:|----------:|----------------:|--------|
| Random baseline | 0.036 | 0.0 | 0.476 |  |
| Markov baseline | 0.148 | 0.052 | 0.734 |  |
| Task 1 — LSTM AE | 0.07 | 0.007 | 0.675 | final_loss=1.933 |
| Task 2 — VAE | 0.046 | 0.002 | 0.826 | final_loss=2.641 |
| Task 3 — Transformer | 0.21 | 0.151 | 1.119 | final_loss=1.784, perplexity=5.96 |
| Task 4 — RLHF | 0.235 | 0.164 | 1.183 | final_reward_pre=0.595, final_reward_post=0.632 |