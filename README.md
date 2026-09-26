# brier

A small model that answers typed questions about a piece of text, reports a
confidence that means what it says, and refuses to answer when it does not know.

Named after the Brier score, because the point of the project is the score, not
the size of the model.

## What it is

Generating text and making a decision are different jobs. Asking a large
language model "is this passage relevant, yes or no?" spends a full
autoregressive forward pass to produce one token, and the probability attached
to that token is not a probability anyone measured. For a pipeline that needs
six judgements about the same paragraph, both of those are the wrong shape.

So this is the other shape:

- **One scorer for every question type.** A boolean is a choice between two
  options, a rubric is a choice between its levels, a plain choice is a choice
  between whatever the caller listed. `f(state, prompt, option) -> score`, with
  a softmax over the options, answers all three. Adding a question type costs no
  new model and no new head.
- **Confidence fitted, then measured.** A temperature is fitted on held-out
  data after training, and the expected calibration error, Brier score and
  reliability curve are reported. Numbers below, including the ones that are
  not flattering.
- **Abstention is a first-class answer.** Below a confidence threshold the
  answer comes back marked `abstained`, and every accessor that would hand the
  caller a usable value raises instead of quietly returning the option the model
  was leaning towards.
- **Our weights.** The encoder is a frozen local embedding model; everything
  above it is a scorer written in numpy, trained with an Adam loop written in
  numpy. 344k parameters, trains in seconds on a laptop CPU, no GPU.

## Measured

Trained on four subjects (revenue, headcount, litigation, rainfall) and tested
on two it had never seen (attendance, infections). Reporting accuracy on new
phrasings of familiar subjects would measure memorisation, so the numbers that
matter are the unseen ones.

| | validation (seen subjects) | test (unseen subjects) |
|---|---|---|
| accuracy | 0.997 | 0.893 |
| mean confidence | 0.997 | 0.902 |
| overconfidence | +0.001 | +0.008 |
| ECE | 0.0032 | 0.0892 |
| Brier | 0.0031 | 0.1809 |

Per question family, on the unseen subjects:

| family | accuracy |
|---|---|
| `relevance`: does this passage address this subject? | 1.000 |
| `subject`: which of these is it about? | 1.000 |
| `tone`: how does it read for the business? | 0.660 |

Abstention on the unseen subjects, which is what a caller actually buys:

| threshold | questions answered | accuracy among them |
|---|---|---|
| 0.0 | 100% | 0.893 |
| 0.6 | 95% | 0.916 |
| 0.8 | 89% | 0.910 |
| 0.9 | 68% | 0.951 |

Training the whole thing from an empty directory takes about 15 seconds. A
decision over four questions about one passage takes about 80 ms, nearly all of
it the one embedding call for the passage; each extra question is a matrix
multiply per option.

## Two findings worth keeping

**A fitted temperature belongs to a domain, not to a model.** Carried from
validation to the unseen subjects, the same weights went from +0.001
overconfidence to **+0.111**, with ECE at 0.1113. Refitting the temperature on
half the unseen subjects brought it back to +0.008 and ECE 0.0892. So a
published calibration number is only a promise about the distribution it was
measured on, and shipping one without saying where it was fitted is the same
mistake as shipping an uncalibrated confidence.

**An aggregate accuracy hides the family that is broken.** An earlier feature
set scored 0.70 overall on the unseen subjects, which looked like a working
model. Broken down: `subject` 1.000, `tone` 0.570, `relevance` **0.500**, exact
chance on a yes/no question. The cause was a design error. For a question whose
options carry content, the state-option comparison decides it; for `"no"`
against `"yes"` the options carry nothing, so the only thing separating them is
how the state answers the prompt, and that interaction was not in the features
at all. The model could learn a prior over the words "no" and "yes" and nothing
else. Adding the state-prompt alignment took `relevance` from 0.500 to 1.000 and
the overall figure from 0.70 to 0.893. The aggregate was never going to say
that; the breakdown said it immediately, and `by_name` is now part of every run.

`tone` at 0.660 is the honest remaining gap. Deciding whether a fact is good
news needs polarity (more litigation is bad, fewer infections is good) and that
does not transfer from four subjects to two new ones the way semantic alignment
does.

## Install and run

Needs Python 3.13 and [Ollama](https://ollama.com) with an embedding model:

```bash
ollama pull nomic-embed-text

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Train, which prints every number in the tables above:

```bash
python scripts/train_model.py
```

Serve it:

```bash
uvicorn brier.api:app --reload
```

```bash
curl -s localhost:8000/decide -H 'content-type: application/json' -d '{
  "state": "According to the filing, revenue grew to 94.9 billion dollars.",
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

In Python:

```python
from pathlib import Path

from brier.decide import Decider
from brier.encoder import OllamaEncoder
from brier.model import load
from brier.types import boolean, score

decider = Decider(encoder=OllamaEncoder(), parameters=load(Path("models/brier.npz")))

answers = decider.decide(
    "According to the filing, revenue grew to 94.9 billion dollars.",
    (
        boolean("relevance", "Does the passage state something about quarterly revenue?"),
        score("tone", "How does the passage read?", ("negative", "neutral", "positive")),
    ),
)

for answer in answers:
    if answer.abstained:
        print(f"{answer.name}: not confident enough ({answer.confidence:.2f})")
    else:
        print(f"{answer.name}: {answer.value} ({answer.confidence:.2f})")
```

## Layout

```
src/brier/
  types.py        Question and Answer; one shape for bool, score and choice
  encoder.py      the frozen embedding model, and a disk cache for it
  features.py     how (state, prompt, option) becomes one row of numbers
  model.py        our weights, the forward pass and the gradients
  train.py        the Adam loop, early stopping, per-family breakdown
  calibration.py  temperature scaling, ECE, Brier, reliability, coverage
  decide.py       the entry point: state in, calibrated answers out
  api.py          HTTP, one request per state
  tasks.py        the synthetic training tasks
scripts/
  train_model.py  train, calibrate, measure, save
```

## What this is not

The training data is synthetic and small: six subjects, a handful of phrasings
each. That is enough to show the machinery works, to catch a design error in the
features, and to demonstrate that calibration does not survive a domain change.
It is not enough to claim the model is good at anything in particular. The next
honest step is real labelled decisions from a real pipeline, where the question
that matters is the one this model already answers best: does this retrieved
passage actually address what was asked?

## Licence

MIT
