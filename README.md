# Steerability of Instrumental-Convergence Tendencies in LLMs

Minimal code to reproduce the `InstrumentalEval` prompt-steering experiments from the paper **“Steerability of Instrumental-Convergence Tendencies in LLMs”**: https://arxiv.org/abs/2601.01584v1  
Based on `InstrumentalEval`: https://github.com/yf-he/InstrumentalEval

We measure how strongly short prompt suffixes can *amplify* or *suppress* instrumental-convergence markers (e.g., shutdown avoidance, self-replication) across Qwen3 variants. Under this protocol, an anti-instrumental suffix sharply reduces convergence rates in aligned models (e.g., Qwen3-30B Instruct: 81.69% → 2.82%), highlighting a safety–security tension: the same steerability that enables safer behavior can also enable adversarial elicitation.

## Quickstart
1. Install dependencies:
   ```bash
   pip install transformers openai hf_transfer accelerate
   ```
2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="..."
   ```
3. Run:
   ```bash
   python main_jh2.py
   ```

## Configure
Edit `ModelConfig` in `main_jh2.py` (HF model, steering mode, sampling, token limits). Outputs are written to `./results/`.
