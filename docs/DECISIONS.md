# DECISIONS LOG

Short records of choices. Do not silently reverse one: add a new entry that supersedes it and say why.

**D1 — Software-only, no hardware.** An earlier attendance-system idea was dropped because it needed physical cameras and could not be prototyped honestly. This project runs on a laptop plus free Colab.

**D2 — Synthetic testbed, labelled as such.** Real MCA/MyGov comments are confidential (the module keeps comments visible only to MCA's internal team), so they cannot be scraped. Training uses public analogues (SemEval-2016) plus a synthetic bill and comments the owner generates and controls. Consequence: results on synthetic data are optimistic; always report SemEval and the hand-labelled gold set alongside it.

**D3 — Stance labels are FAVOR / AGAINST / NONE.** This matches SemEval-2016 Task 6, so Stage 1 transfers directly and numbers are comparable to published work. The deck (slides 9 and 12) says "support/oppose/conditional". **"Conditional" is not modelled** and is listed as future work. The deck must be corrected (see PROGRESS → Open issues).

**D4 — Two-stage training.** Stage 1 on SemEval, Stage 2 on synthetic train. The Stage 1-zero-shot vs Stage 2 gap is the domain-adaptation ablation.

**D5 — AI-text detector must be HC3-style, not a GPT-2 detector.** An early note suggested `roberta-base-openai-detector`; that model was built to detect GPT-2 output, which is the wrong target for modern LLM text. Use a RoBERTa classifier trained on HC3 (Human-ChatGPT Comparison Corpus), **after verifying the model id and licence on Hugging Face**. Treat output as a weak flag, never an accusation.

**D6 — SQLite for the prototype, PostgreSQL as the production note.** The slides list PostgreSQL. A single-user demo needs no server; SQLite removes setup risk. Say "PostgreSQL for production deployment" if asked, or update slide 11.

**D7 — Three duplicate detectors, compared.** Exact, MinHash-LSH and SBERT-semantic are all implemented so the paper can show where exact matching fails on paraphrased campaigns. This is the project's most defensible novelty claim, so it must be measured, not asserted.

**D8 — Colab for training only; the app runs locally.** Colab sessions are ephemeral and cannot host a demo. Weights travel Colab → Drive/HF Hub → local `models/`.

**D9 — The repo is the memory.** AGENTS.md + docs/ carry all context so the owner can switch AI agents when credits expire. Every session ends with a PROGRESS.md update.

**D10 — Campaign families never split across train/dev/test.** Otherwise near-duplicates leak between splits and inflate scores.

**D11 — Hindi/multilingual is a stretch goal.** The deck says "planned"; keep it off the critical path (IndicTrans2 in front of the English pipeline, only if time remains).

**D13 — Stage 1 data comes from the TweetEval packaging of SemEval-2016 Task 6A, loaded from Hugging Face** (`cardiffnlp/tweet_eval`, configs `stance_abortion|atheism|climate|feminist|hillary`). Verified sizes: train 2,620 / validation 294 / test 1,249 (4,163 labelled tweets in total; an earlier note saying ~4,870 was wrong). It needs no manual download, which removes a failure point in copy-paste Colab. Differences from the official release: five of the targets only, TweetEval's own preprocessing, and the Hub lists its licence as "unknown", so **cite both SemEval-2016 (Mohammad et al.) and TweetEval (Barbieri et al., 2020, arXiv 2010.12421)** and describe it honestly as a research benchmark. Report macro-F1 plus SemEval's official F_avg (mean of FAVOR and AGAINST F1), noting that our numbers are not strictly comparable to the official leaderboard.

**D14 — Label ids follow TweetEval: 0 = NONE, 1 = AGAINST, 2 = FAVOR.** Use this order everywhere (training, synthetic data, API, dashboard). Model input is a sentence **pair** `(target, text)` encoded by the tokenizer, not a string with a literal "[SEP]". For legislation, target = clause text and text = comment.

**D15 — Training is built first, in Colab, by copy-paste cells (`notebooks/COLAB_STAGE1.md`); the rest of the app is built afterwards with AI coding agents (Antigravity, Cline).** Training cells use a plain `torch.utils.data.Dataset` and sklearn metrics. Do **not** reintroduce `datasets.Dataset.map/from_pandas` or the `evaluate` library in training code: `from_pandas`/`map` raised a dill pickling error in the test environment (Python 3.13, datasets 5.1.0), and the plain-torch route is version-proof. Do not pass `tokenizer=` to `Trainer` (removed in recent transformers); the cells use `DataCollatorWithPadding` instead and handle the `eval_strategy` / `evaluation_strategy` rename.

**D12 — Cited sources are limited to verified ones.** The reference list in the deck has seven checked sources. An earlier "IJERT 2026" citation could not be verified and was replaced by the Team Qbit SIH25035 project page.

**D16 — Paper scope and the synthetic dedupe benchmark (2026-10-21).** The conference paper is scoped to: (a) the Stage 1 stance baseline (SemEval/TweetEval) plus its zero-shot behaviour on the synthetic data, and (b) a comparison of exact vs MinHash vs SBERT duplicate detection on a synthetic consultation benchmark with planted campaigns. Stage 2, the AI-text detector and the summariser are future work. The benchmark (`data/synthetic/`): a fictional 6-clause "Companies (Amendment) Bill, 2026"; 240 organic comments (40 per clause: 14 FAVOR / 14 AGAINST / 12 NONE); 8 campaign families of 15 comments each (5 verbatim with surface edits, 5 light word-level edits, 5 heavy rewrites). Total 360. **All text was authored by an LLM (Claude) inside a coding session, not by real commenters**; the paper must say so. Splits: campaigns K1-K2 train, K3-K5 dev, K6-K8 test (never straddling, D10); organic comments stratified per clause and stance (train 114 / dev 66 / test 60). Thresholds are tuned on dev by pairwise F1 (middle of any tied range); test is evaluated once. Known construction limits: heavy rewrites share almost no 3-shingles with their template by design (Jaccard ~0), so MinHash failing on them is expected rather than a discovery; the interesting result is SBERT's precision against organic comments on the same clause and stance. The 60-comment gold set (organic test split) is hand-labelled by the owner and compared with generator labels via `data/eval/score_gold.py`. `is_machine_generated` is False for every row because AI-text detection is out of the paper's scope.
