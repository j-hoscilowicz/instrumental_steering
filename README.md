# The Safety–Security Dilemma of Modern AI Systems

Minimal code to reproduce the `InstrumentalEval` prompt-steering experiments. Based on `InstrumentalEval`: https://github.com/yf-he/InstrumentalEval

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
