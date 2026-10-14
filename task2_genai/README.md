# Task 2 — Supply-chain anomaly explainer (QLoRA)

<a href="https://colab.research.google.com/github/YOUR_GITHUB/CDAZZDEV-MLE-YourName/blob/main/task2_genai/notebooks/supply_chain_finetune.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- **Merged model:** `https://huggingface.co/<hf-user>/supply-chain-anomaly-qwen2.5-3b-instruct` (printed by the last notebook cell)
- **Teacher / student:** Groq `openai/gpt-oss-120b` labels the data; `Qwen/Qwen2.5-3B-Instruct` is fine-tuned. The account no longer serves Llama 3.3 70B, and the teacher stays a different family from the student.

## Problem
Input is one control-tower incident record (event type, industry, quantities, dwell, optional temperature and supplier on-time rate, operator note). Output is JSON: `likely_cause`, `supporting_signals`, `immediate_action`, `escalation`, `residual_risk`, `confidence`. A correct answer cites only facts in the record and gives a concrete action. Inventing metrics, contradicting the numbers, or a generic "investigate" action counts as incorrect.

## Files
| Path | Purpose |
| --- | --- |
| `src/scenarios.py` | 150-incident grid: 15 event types x 10 variants, 6 industries, 3 prompt formats |
| `src/generate_dataset.py` | Groq teacher labelling, Pydantic validation, stratified 80/10/10 split |
| `src/diversity.py` | Length stats, topic frequencies, near-duplicate rate (token Jaccard >= 0.8) |
| `src/evaluate.py` | ROUGE-L, LLM-as-judge, hallucination rate, confidence parsing |
| `prompts/` | Teacher, student, and judge system prompts |
| `sop_corpus/sops.jsonl` | SOP snippets for the ChromaDB RAG fallback |
| `data/` | `train/val/test.jsonl` (Qwen chat format) + `*_meta.jsonl` + diversity report |
| `notebooks/supply_chain_finetune.ipynb` | Training, evaluation, RAG fallback, merge + push (run on Colab) |

## Run
```bash
python -m task2_genai.src.generate_dataset --dry-run   # diversity only, no API calls
python -m task2_genai.src.generate_dataset             # needs GROQ_API_KEY; writes data/*.jsonl
```
Commit `data/`. In the notebook, set `REPO_URL` and add the `GROQ_API_KEY` and `HF_TOKEN` Colab secrets. Run all cells on a T4 or L4 GPU, fill in `REVIEW_LABELS` and the analysis paragraphs, then download the notebook with outputs and commit it.

Split: 120 train / 15 val / 15 test (stratified by event type).
