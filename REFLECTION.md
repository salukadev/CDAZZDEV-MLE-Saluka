# Reflection

## Architectural decisions

**Task 2.** I picked supply-chain anomaly explanation because correctness can be checked: every claim in the answer must trace back to a field in the incident. The dataset is built from a scenario grid rather than free-form teacher generation. The grid has 15 event types, 10 variants each, 6 industries, and 3 input formats (prose, JSON extract, shift log). Each variant draws a different subset of floor notes, so the dataset is diverse by construction and the diversity report proves it: each event type is 6.7% of the set and no prompt pair reaches 0.8 token Jaccard. The teacher (`openai/gpt-oss-120b` on Groq; Llama 3.3 70B was not offered on this account) only writes labels. Every label is validated against a Pydantic schema before it enters the dataset. The student is Qwen2.5-3B, a different model family, small enough for QLoRA on a free T4. The base and fine-tuned models are compared on the same 4-bit weights by toggling the adapter, so the comparison isolates the effect of fine-tuning. Besides ROUGE-L, I report schema-valid JSON rate and an LLM judge. ROUGE alone rewards wording overlap, not faithfulness.

**Task 3.** I used LangGraph for both parts:
- **3A:** a ReAct agent picks its own tools. A forced empty news result makes the observe-and-replan step visible.
- **3B:** an explicit graph, because the critique loop must run every time. The model writes the analysis and the report. Tool access is enforced in code, so a cross-role call raises an error rather than relying on the prompt.
- **Handoffs** are Pydantic models (`DataBrief`, `ClarificationRequest`, `ClarificationResponse`, `ResearchReport`).
- **Errors:** every tool returns `{"ok": false, "error": ...}` instead of raising, so each agent has a defined alternative: a shorter price history, keeping the 30-day volatility figure if the 90-day one fails, web search instead of news, or a rule-based report if the LLM output fails validation.
- **Critique question:** the 30-day volatility window does not match the 90-day horizon, so the clarification changes the hedge sizing.

## What I would improve with more time
- Run the 3B agents as LLM tool-callers inside their restricted tool sets, not as fixed node logic. The orchestration would become as autonomous as 3A while keeping the code-level permission check.
- Compute training loss on assistant tokens only. The Qwen2.5 chat template has no generation markers, so the loss currently also covers the system and user turns.
- Add contrastive training examples: records that look like one event type but are another (a late but complete trailer versus a shortage). These target the most likely confusions.
- Use several annotators for the hallucination review, and add per-field accuracy (escalation flag, cause class) alongside ROUGE.
- Use an implied-volatility source for the hedge instead of historical volatility.

## Limitations
- The 150 labels come from one teacher, so the student inherits its blind spots. The judge is the same model family as the teacher, which can bias it toward teacher-style answers.
- With 15 test rows, metric differences are directional, not statistically significant.
- yfinance and DuckDuckGo are unofficial and rate-limited. The tools degrade instead of failing, but report quality drops when they return nothing.
- The persistent cache is keyed by ticker and calendar day, so a rerun on the same day never refreshes prices.
