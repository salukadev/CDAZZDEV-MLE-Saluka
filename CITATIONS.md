## CitationsTeacher and judge model (dataset labels and LLM-as-judge scores, not the code): **Groq** `openai/gpt-oss-120b`. The full teacher system prompt is in the appendix. It is also printed from `task2_genai/prompts/teacher_system.txt` in `task2_genai/notebooks/supply_chain_finetune.ipynb`.

No third-party source file was copied. Indicator formulas follow the published definitions (Wilder RSI, MACD as the difference of two EMAs, Bollinger Bands with population standard deviation) and were written in `task3_agentic/src/indicators.py`. Libraries (`transformers`, `peft`, `trl`, `bitsandbytes`, `langgraph`, `yfinance`, `ddgs`, `chromadb`) are dependencies, not adapted source.

## Task 2 — supply-chain fine-tune


| File                                     | Functions                                                                                                              |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `task2_genai/src/scenarios.py`           | `_metrics`, `_render`, `build_scenarios`                                                                               |
| `task2_genai/src/generate_dataset.py`    | `load_prompt`, `extract_json`, `label_one`, `stratified_split`, `write_jsonl`, `generate`, `main`                      |
| `task2_genai/src/diversity.py`           | `tokens`, `jaccard`, `length_stats`, `percentile`, `frequency`, `max_share`, `near_duplicate_rate`, `diversity_report` |
| `task2_genai/src/evaluate.py`            | `mean_rouge_l`, `judge_one`, `hallucination_rate`, `parse_confidence`                                                  |
| `task2_genai/src/schemas.py`             | `AnomalyExplanation`, `JudgeScore`                                                                                     |
| `task2_genai/prompts/teacher_system.txt` | teacher system prompt (full text in the appendix)                                                                      |
| `task2_genai/prompts/student_system.txt` | student system prompt                                                                                                  |
| `task2_genai/prompts/judge_system.txt`   | judge system prompt                                                                                                    |
| `task2_genai/sop_corpus/sops.jsonl`      | SOP snippets for the ChromaDB fallback                                                                                 |
| `task2_genai/README.md`                  | task write-up                                                                                                          |


`task2_genai/data/train.jsonl`, `val.jsonl`, and `test.jsonl` assistant turns are labels from Groq `openai/gpt-oss-120b` via `label_one`, validated with `AnomalyExplanation`. The user prompts in those files come from `build_scenarios`.

## AI-ASSISTED: Grok (Cursor), Prompt: 'Fix the Colab QLoRA notebook: token-length TypeError, TRL chunked-logprob RuntimeError, GPU not used, and PeftModel.merge_and_unload import error. Show only the cell changes to apply on the running notebook.', Date: 2026-10-13


| File                                                | Functions                                 |
| --------------------------------------------------- | ----------------------------------------- |
| `task2_genai/notebooks/supply_chain_finetune.ipynb` | `train`, `generate`, `valid`, `summarise` |


The same notebook's training, evaluation, RAG-fallback, and merge-and-push cells were drafted on 2026-10-13 under the primary prompt above, then patched the same day for those runtime errors.

## Task 3 — agentic research



## AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the given architecture for a supply-chain fine-tune and a LangGraph research system completing the incomplete functions and following the provided rules', Date: 2026-10-13


| File                                            | Functions                                                                                                                                                                                                                                            |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task3_agentic/src/indicators.py`               | `sma`, `rsi_wilder`, `macd`, `bollinger_bands`, `attach_indicators`, `annualized_volatility`                                                                                                                                                         |
| `task3_agentic/src/tools.py`                    | `arm_empty_news_once`, `_traced`, `_num`, `_price`, `_news`, `_volatility`, `_sentiment`, `_search`, `get_price_data`, `get_news`, `calculate_volatility`, `llm_sentiment`, `web_search`, `call_tool`                                                |
| `task3_agentic/src/tracing.py`                  | `set_agent`, `log_tool_call`, `log_validation_failure`, `count_trace_lines`                                                                                                                                                                          |
| `task3_agentic/src/memory.py`                   | `brief_path`, `save_brief`, `load_brief`                                                                                                                                                                                                             |
| `task3_agentic/src/graph.py`                    | `_use`, `_groq_json`, `quantitative_risks`, `writer_gather`, `analyst_brief`, `writer_review`, `analyst_clarify`, `_fallback_report`, `writer_report`, `build_multi_agent_graph`, `run_multi_agent`, `build_single_agent`, `ask`, `run_single_agent` |
| `task3_agentic/src/pipeline.py`                 | `print_messages`, `print_report`, `main`                                                                                                                                                                                                             |
| `task3_agentic/src/schemas.py`                  | `HeadlineSentiment`, `SentimentResult`, `PriceSnapshot`, `VolatilityResult`, `RiskFlag`, `DataBrief`, `ClarificationRequest`, `ClarificationResponse`, `ResearchReport`                                                                              |
| `task3_agentic/src/prompts.py`                  | `SINGLE_AGENT_SYSTEM`, `ANALYST_NOTES_SYSTEM`, `WRITER_SYSTEM`, `SENTIMENT_SYSTEM`, `CLARIFICATION_QUESTION`                                                                                                                                         |
| `task3_agentic/app/dashboard.py`                | `load_rows`, `main`                                                                                                                                                                                                                                  |
| `task3_agentic/notebooks/research_agents.ipynb` | notebook that runs the single-agent and two-agent pipelines                                                                                                                                                                                          |
| `task3_agentic/README.md`                       | task write-up                                                                                                                                                                                                                                        |


`llm_sentiment` and the analyst/writer nodes call Groq for sentiment, notes, and the report. Those calls are application use of the model, separate from the Cursor drafting of the functions.

## Shared files



## AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the given architecture for a supply-chain fine-tune and a LangGraph research system completing the incomplete functions and following the provided rules', Date: 2026-10-13

- `README.md`
- `REFLECTION.md`
- `requirements.txt`
- `requirements-colab.txt`
- `.gitignore`
- `.env.example`



## Appendix — teacher system prompt

Used by `label_one` in `task2_genai/src/generate_dataset.py` with Groq `openai/gpt-oss-120b`. Source file: `task2_genai/prompts/teacher_system.txt`.

```
You are a senior supply-chain control-tower analyst writing gold labels for a fine-tuning dataset. You will be given one incident record. Return only a JSON object, with no markdown and no prose around it.

The JSON object must contain exactly these keys:
- likely_cause: string. Name one operational mechanism that the record supports. Do not merely repeat the event type.
- supporting_signals: array of 2 to 4 strings. Each string must cite a fact that appears in the record, including the number or condition when one was given.
- immediate_action: string. One specific next action a planner or warehouse lead can execute today. Name the object of the action (the SKU, lane, dock, carrier, or supplier) when the record provides it.
- escalation: boolean. True when the record shows a safety limit breach, a regulated product at risk, a customer line-down, a hospital or store-critical shortage, or high severity. Otherwise false.
- residual_risk: string. What can still go wrong after the immediate action, limited to risks implied by the record.
- confidence: number from 0 to 1. Use a value below 0.6 when a missing field leaves two causes equally plausible. Use 0.6 or above when the cited signals distinguish one cause.

Rules:
- Use only facts that appear in the incident. Do not invent costs, people, carriers, temperatures, or on-time rates.
- If a field is absent, do not assume its value.
- Do not answer with only "investigate further". The immediate action must change the physical or planning state of the incident.
- If quantity observed is far below quantity expected, do not claim the shipment is complete.
- If temperature is inside 2 to 8 Celsius, do not call it a cold-chain excursion.
- If temperature is above 8 Celsius on a cold-chain move, treat temperature as a primary signal.
```

