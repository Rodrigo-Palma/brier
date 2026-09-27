# brier

A small model that answers typed questions about a piece of text, reports a
confidence, and refuses to answer when it does not know.

Named after the Brier score, because the project is about the score, not the
size of the model. The numbers below include the ones that do not flatter it.

```
                    ┌───────────────────────────────────────────┐
  "In the year      │  frozen encoder (not ours, never trained)  │
   under review,    │  nomic-embed-text, 768 dimensions          │
   revenue grew     └───────────────────┬───────────────────────┘
   to 94.9bn."                          │  one vector per text
                                         ▼
  "Does this        ┌───────────────────────────────────────────┐
   passage state    │  7 feature blocks per option              │
   something        │  state · option · state⊙option · |s−o|    │
   about revenue?"  │  prompt⊙option · s⊙p⊙o · (s·p)o           │
                    └───────────────────┬───────────────────────┘
  options:                               ▼
  ["no", "yes"]     ┌───────────────────────────────────────────┐
                    │  OUR WEIGHTS: 5376 → 64 → 1, numpy        │
                    │  344k parameters, 15s to train on CPU     │
                    └───────────────────┬───────────────────────┘
                                         ▼
                    ┌───────────────────────────────────────────┐
                    │  softmax over the options ÷ temperature    │
                    │  temperature fitted on UNSEEN subjects     │
                    └───────────────────┬───────────────────────┘
                                         ▼
                      yes, 0.83   │   or: abstained
                                  │   (accessors raise, they do
                                  │    not hand back a guess)
```

## The idea

Generating text and making a decision are different jobs. Asking a language
model "is this passage relevant, yes or no?" spends a full autoregressive pass
to produce one token, and the probability attached to that token is not a
probability anyone measured.

So this is the other shape. **Every question is a choice between options
described in words.** A boolean is a choice between two, a rubric is a choice
between its levels, a plain choice is a choice between whatever the caller
listed. One scorer `f(state, prompt, option)` with a softmax over the options
answers all three, so a new question type costs no new model and no new head.

The encoder underneath is frozen and borrowed. Everything above it — the
forward pass, the gradients, the Adam loop, the calibration — is ours, in numpy.

## What the measurements say

Four subjects for training, two more for fitting the temperature, two more for
testing. **No subject appears in more than one split, and no row appears
twice**: `generate` refuses to repeat a question across splits and fails loudly
when the space runs out.

### The headline is that the model does not beat the encoder

| on the test subjects (n=200 distinct) | accuracy |
|---|---|
| this model, 344k parameters | 0.820 [0.761, 0.867] |
| cosine baseline, ≤1 parameter | 0.815 [0.755, 0.863] |
| **difference** | **+0.005 [−0.030, +0.040]** |

The baseline is `argmax cos(state, option)` for questions whose options carry
content, and a threshold on `cos(state, prompt)` for questions whose options do
not, with that threshold fitted on the training split. Three seeds: +0.005,
−0.020, +0.020. On the calibration subjects the model is **worse**, by −0.095
[−0.160, −0.035].

Per family, model against baseline:

| family | model | cosine | majority |
|---|---|---|---|
| `relevance` — does the passage address this subject? | 1.000 | 1.000 | 0.540 |
| `subject` — which of these is it about? | 1.000 | 1.000 | 0.520 |
| `answerable` — does it answer *this* question? | 0.720 | **0.800** | 0.500 |
| `tone` — how does it read for the business? | **0.560** | 0.460 | 0.520 |

`tone` is the only place the trained head earns its parameters, and it is the
only task that needs something similarity cannot give: polarity. More litigation
is bad news, fewer infections is good news, and neither is a matter of which
words are nearby.

### One level below the family average, a subtype sits at zero

`answerable` reads 0.720. Broken down by which part of the question was wrong:

| subtype | accuracy |
|---|---|
| nothing wrong (answerable) | 1.000 [0.867, 1.000] n=25 |
| wrong subject | 0.750 [0.409, 0.929] n=8 |
| wrong company | 0.625 [0.306, 0.863] n=8 |
| **wrong year** | **0.000 [0.000, 0.299] n=9** |

Zero. The model answers "yes" to every question about the wrong year, and it
does so at above-average confidence, so **abstention does not filter this
failure**: at threshold 0.7 every surviving error is a wrong-year one. A single
token inside a mean-pooled vector is not something a head on top can recover,
and this is the shape of that ceiling.

### The calibration number is inside its own noise floor

| | test subjects |
|---|---|
| ECE, 10 fixed bins over [0, 1] | 0.0657 |
| what a **perfectly calibrated** model scores at this n | 0.0546 [0.0258, 0.0857] |
| verdict | inside the floor: no evidence of miscalibration |

ECE is biased upward and the bias grows as the sample shrinks, so at n=200 a
flawless model still scores about 0.055. Publishing 0.0657 as evidence of good
calibration would be reading noise. Every report prints the floor beside the
value and says which side of it the measurement is on.

The bins are fixed over [0, 1] because an earlier version spread them over the
range the model actually used, and that scored **0.0899 where fixed bins score
0.0371 on the same predictions**.

### What does survive: the confidence ranks

| on the test subjects | |
|---|---|
| AUROC (confidence vs correctness) | **0.767 [0.678, 0.849]** |
| permutation test | **p = 0.0005** |
| area under the risk-coverage curve | 0.0742 |

This is one number with an interval instead of five correlated rows, and it is
what abstention actually rests on. It holds across all three seeds.

| abstain below | answers | accuracy among them |
|---|---|---|
| — | 100% | 0.820 [0.761, 0.867] |
| 0.6 | 89.0% | 0.865 [0.807, 0.908] |
| 0.7 | 72.5% | 0.890 [0.828, 0.931] |
| 0.8 | 50.5% | 0.931 [0.864, 0.966] |
| 0.9 | 18.5% | 0.946 [0.823, 0.985] |

**No single row of this table proves anything on its own** — the intervals
overlap the no-abstention rate. The AUROC above is the claim; this table is the
illustration of it.

### Cost

| | with the encoder live | with vectors cached |
|---|---|---|
| 1 question | 34.6 ms | 0.13 ms |
| 4 questions | 52.3 ms | 0.16 ms |
| 16 questions | 107.4 ms | 1.56 ms |

Almost all of it is the encoder. The part that is ours costs tenths of a
millisecond, which is the economy of the design: the state is encoded once and
each extra question is a matrix multiply per option.

## Three things worth keeping

**A fitted temperature belongs to a domain, not to a model.** Carried from the
calibration subjects to the test subjects, the same weights move from −0.002 to
+0.105 of overconfidence. The temperature is now fitted on subjects the weights
never saw, precisely so this cannot be blamed on having fitted it in-domain.

**An aggregate hides the family that is broken, and a family hides its own
subtypes.** An earlier feature set scored 0.70 overall, which looked like a
working model; `relevance` was at exactly 0.500, chance on a yes/no question.
The cause was algebraic: when the options are `"no"` and `"yes"` they carry no
content, so no feature built from (state, option) or (prompt, option) alone can
separate them — only the three-way interaction can. `by_name` and `by_tag` are
now part of every run, and `by_tag` is what found the wrong-year zero above.

**A frozen encoder is a hard ceiling, and measuring the encoder is how you find
it.** On Ollama 0.18.0 with `nomic-embed-text`, every capitalised token collapses
onto one vector: `cos("Apple", "Cat") = 1.0000` while `cos("Apple", "apple") =
0.4706`. Two sentences differing only in a company name came back byte for byte
identical, and 287 distinct texts produced 119 distinct vectors. This is a known
regression, [ollama/ollama#15609](https://github.com/ollama/ollama/issues/15609),
whose root cause is `BasicTokenizer` preprocessing lost in the HF→gguf
conversion; the issue frames it as a non-ASCII problem, and it is wider than
that — **18.3% of the words in a real 10-K start with a capital**. Lower-casing
before encoding is one line and it is why the numbers above are what they are.

## Install and run

Needs Python 3.13 and [Ollama](https://ollama.com):

```bash
ollama pull nomic-embed-text

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/train_model.py     # prints every number above, about 15 seconds
```

Serve it:

```bash
uvicorn brier.api:app --reload

curl -s localhost:8000/ready
curl -s localhost:8000/decide -H 'content-type: application/json' -d '{
  "state": "In the year under review, revenue grew to 94.9 billion dollars.",
  "questions": [
    {"name": "relevance", "kind": "bool",
     "prompt": "Does the passage state something about quarterly revenue?",
     "options": ["no", "yes"]},
    {"name": "tone", "kind": "score",
     "prompt": "How does the passage read for the business?",
     "options": ["negative", "neutral", "positive"]}
  ]
}'
```

Both questions are answered from one encoding of the passage.

```python
from pathlib import Path

from brier.decide import Decider
from brier.encoder import OllamaEncoder
from brier.model import load
from brier.types import Abstained, boolean

decider = Decider(encoder=OllamaEncoder(), parameters=load(Path("models/brier.npz")))
answer = decider.decide(
    "In the year under review, revenue grew to 94.9 billion dollars.",
    (boolean("relevance", "Does the passage state something about revenue?"),),
)[0]

try:
    print("yes" if answer.as_bool else "no", f"({answer.confidence:.2f})")
except Abstained as declined:
    print(declined)          # reading a value it did not commit to raises
```

## Layout

```
src/brier/
  types.py        Question and Answer; one shape for bool, score and choice
  encoder.py      the frozen encoder, and a training cache with its own warnings
  features.py     how (state, prompt, option) becomes one row of numbers
  model.py        our weights, forward pass, gradients, provenance
  train.py        the Adam loop, early stopping, breakdown by family and subtype
  baseline.py     what the encoder answers with at most one parameter
  calibration.py  temperature, ECE with its noise floor, reliability, coverage
  statistics.py   Wilson intervals, AUROC, AURC, the ECE floor
  evaluate.py     model beside baseline, with the interval for the difference
  decide.py       state in, calibrated answers out
  api.py          HTTP: one request per state, /ready reports the weights
  tasks.py        the synthetic tasks, hard negatives included
```

99 tests, 98% coverage, and the suite runs offline with a deterministic fake
encoder.

## What this is not

The training data is synthetic and the space is small: eight subjects, a dozen
phrasings each, and the splits are sized to what the generator can actually
produce without repeating itself. That is enough to show the machinery works, to
catch two design errors in the features, and to demonstrate that calibration
does not survive a change of subject. It is not enough to claim the model is
good at anything.

Specifically not claimed: that 344k trained parameters beat the frozen encoder
(measured: they do not), that the calibration is good (measured: the error is
inside its own noise floor), or that abstention protects against the model's
failures (measured: it does not protect against the wrong-year one). What is
claimed is that the confidence ranks correctness, with an interval and a p-value,
and that every number here comes with the n it was measured on.

The next honest step is real labelled decisions rather than a wider generator.

## Licence

MIT
