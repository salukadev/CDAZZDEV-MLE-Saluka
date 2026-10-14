# CDAZZDEV MLE Assessment — Tasks 2 and 3

| Task | Folder | Deliverable |
| --- | --- | --- |
| Task 2 — Generative AI fine-tuning | [`task2_genai/`](task2_genai/README.md) | Supply-chain anomaly explainer, QLoRA on Qwen2.5-3B, Colab notebook, merged model on Hugging Face |
| Task 3 — Agentic workflows | [`task3_agentic/`](task3_agentic/README.md) | LangGraph single agent + two-agent critique loop, memory, `logs/agent_trace.jsonl` |

Also see [`CITATIONS.md`](CITATIONS.md) and [`REFLECTION.md`](REFLECTION.md).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY (and HF_TOKEN for Task 2)
```

Keys are read from `.env` locally and from Colab Secrets in Colab. No credential is stored in this repository.

## Quick commands

```bash
# Task 2: check prompt diversity, then label the dataset with the Groq teacher
python -m task2_genai.src.generate_dataset --dry-run
python -m task2_genai.src.generate_dataset

# Task 3: run the two-agent pipeline (second run on the same day loads the cache)
python -m task3_agentic.src.pipeline --ticker AAPL --ignore-cache
python -m task3_agentic.src.pipeline --ticker AAPL
streamlit run task3_agentic/app/dashboard.py
```
