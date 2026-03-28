# Corpus scaling: simple stories and vocabulary

The default [`data/corpus.txt`](../data/corpus.txt) is only **dozens of lines**—enough to smoke-test [`sandbox.py`](../sandbox.py), not to learn fluent text. For coherent generations, aim for **thousands to tens of thousands** of lines (one short sentence per line, UTF-8). This guide matches how the script actually behaves.

## Why simple stories fit this model

Training uses **sliding windows** (default **6** tokens) and **trajectory contrastive** loss on **window-level** dynamics (`run_window_dynamics`), not classic transformer attention. Short, clean sentences align with that geometry:

- **Narrative continuity** — cause–effect and repetition-with-variation give stable trajectories to learn.
- **Short lines** — stay inside a small context window and avoid huge OOV drops per line.
- **Recurring wording** — similar attractor basins revisited with mild variation, which matches tension-driven evolution.

This is in the spirit of small-data / **BabyLM**-style setups (the repo does not implement BabyLM benchmarks; it is a dynamics sandbox).

## How vocabulary actually works (read this before writing data)

| Fact | Detail |
|------|--------|
| **Vocab is fixed at import** | `FULL_VOCAB` is built when `sandbox.py` loads: seed lines, unique words from **`data/corpus.txt` only**, then alphabetical cap at **512** and fill from a word blob. |
| **`--corpus` does not add words** | Your training file only uses tokens **already in** `FULL_VOCAB`. Unknown words are **dropped** per line; see the startup **Corpus coverage** line. |
| **To introduce new words** | Add them to `data/corpus.txt` (and/or adjust vocab construction in `sandbox.py`) so they enter the 512 **before** relying on them in a custom corpus. |

Export the current word list anytime:

```bash
python sandbox.py --print-vocab > vocab.txt
```

Use **`vocab.txt`** as an allowlist when preprocessing raw text (see below).

## Line format (required)

Same as [`load_corpus`](../sandbox.py): **one sentence per line**, UTF-8, blank lines skipped, lines starting with **`#`** are comments.

Example:

```text
the fox ran through the tall grass.
# practice sentence
the dog woke up and looked around.
```

The script reports how many lines have at least **`window_size + 1`** in-vocabulary tokens after OOV removal.

## Target scale

A practical first step: **about 5,000–15,000** short sentences (roughly tens to low hundreds of thousands of tokens). That is enough for the attractor to see real patterns on CPU without requiring a large GPU run.

## Preparing a corpus from raw stories

Use [`scripts/text_to_corpus_lines.py`](../scripts/text_to_corpus_lines.py) (stdlib only):

```bash
python sandbox.py --print-vocab > vocab.txt
python scripts/text_to_corpus_lines.py gutenberg_snippet.txt -o data/my_stories.txt --allowlist vocab.txt --min-words 4 --max-words 16
```

- **`--allowlist`** — keeps only words in your exported vocab (drops sentences that would become empty or too short).
- **`--min-words` / `--max-words`** — drop lines outside a length band (defaults 4–24 words after filtering).

Segmentation is a simple **`.?!`** split (no NLTK). Proofread or post-filter noisy lines if needed.

## Training example (copy-paste)

```bash
source .venv/bin/activate
python sandbox.py \
  --corpus data/my_stories.txt \
  --epochs 50 \
  --val-fraction 0.05 \
  --epoch-copies 2 \
  --epoch-metrics-csv metrics.csv \
  --state-dim 512
```

- **`--epochs`** — number of training passes (defaults to the same value as the `NUM_EPOCHS` constant if omitted).
- **`--state-dim`** — embedding / dynamics width (default **512**); increase only if you scale data and need capacity.
- **`--window-size`** — context length (default **6**); longer windows need longer usable lines.

With a **tiny validation split** (e.g. only **2** lines held out), **val CE is noisy**—trust **train CE**, trajectory metrics, and qualitative generations until you have more held-out lines. See [`BASELINE.md`](BASELINE.md).

## Optional: synthetic story prompts (vocab-safe)

Use these as **ideas** for hand-written sentences or for prompting an external model, with the constraint: **only use words from `python sandbox.py --print-vocab`**.

1. The **fox** and the **dog** meet near the **house**.
2. A **clear** **mind** sees the **pattern** in the **system**.
3. **Quick** **thoughts** **jump** over **lazy** **dogs** in the **sun**.
4. The **reason** for the **effect** was **clear** to every **mind**.
5. **Cause** and **effect** **dance** inside the **flow** of **time**.
6. The **system** **demands** a **strong** **solution** **fast**.
7. **Brown** **fox** **jumps** when the **path** is **clear**.
8. **Lazy** **dog** **rests** in the **mind** of the **deep** **system**.
9. **Pattern** **recognition** **happens** in the **mind** **first**.
10. **Reason** **builds** the **bridge** between **cause** and **effect**.
11. The **flow** of **mind** **moves** like a **quick** **stream**.
12. **Every** **clear** **solution** **creates** a **stable** **pattern**.
13. **Pure** **reason** **responds** when the **system** **stays** **calm**.
14. The **dog** **follows** the **clear** **path** **home**.
15. **Strong** **active** **thoughts** **dissolve** **lazy** **fear**.
16. **Nature** **shows** the **true** **effect** of **clear** **mind**.
17. **One** **simple** **pattern** **holds** the **whole** **system**.
18. The **future** **flows** from **deep** **reason** and **care**.
19. **Beautiful** **mind** **understands** **slow** and **fast** **time**.
20. **Lasting** **peace** **comes** when **cause** meets **clear** **effect**.

(Not every word may exist in your current `FULL_VOCAB`—trim or swap using `vocab.txt`.)

## What we do not ship in-repo

Thousands of lines of copyrighted or heavy story text are **not** bundled here (size and licensing). Add your own files under `data/` or elsewhere and pass **`--corpus`**.
