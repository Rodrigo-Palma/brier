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
| accuracy | 0.980 | 0.853 |
| mean confidence | 0.971 | 0.851 |
| overconfidence | -0.009 | -0.002 |
| ECE | 0.0182 | 0.0899 |
| Brier | 0.0395 | 0.2267 |

Per question family, on the unseen subjects:

| family | accuracy |
|---|---|
| `relevance`: does this passage address this subject? | 1.000 |
| `subject`: which of these is it about? | 1.000 |
| `tone`: how does it read for the business? | 0.760 |
| `answerable`: does it answer *this* question, right company and year? | 0.680 |

Abstention on the unseen subjects, which is what a caller actually buys:

| threshold | questions answered | accuracy among them |
|---|---|---|
| 0.0 | 100% | 0.853 |
| 0.6 | 91% | 0.891 |
| 0.7 | 82% | 0.894 |
| 0.8 | 77% | 0.905 |
| 0.9 | 50% | 0.947 |

Training the whole thing from an empty directory takes about 15 seconds. A
decision over four questions about one passage takes about 80 ms, nearly all of
it the one embedding call for the passage; each extra question is a matrix
multiply per option.

Used as the relevance gate of a retrieval pipeline over a real SEC 10-K, against
ten questions of which five the filing can answer and five it cannot, it cut the
questions wrongly admitted from 4 of 5 to 2 of 5 while losing one it should have
admitted. The two it still admits wrongly are both wrong-year questions, which is
exactly where `answerable` measures 0.680.

## Three findings worth keeping

**A fitted temperature belongs to a domain, not to a model.** Carried from
validation to the unseen subjects, the same weights went from -0.009
overconfidence to **+0.105**, with ECE at 0.1153. Refitting the temperature on
half the unseen subjects brought it back to -0.002 and ECE 0.0899. So a
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

**The encoder was throwing away every proper noun, and measuring the data
found it.** The `answerable` family asks whether a passage answers *this*
question: right subject, right company, right year. It sat at chance, and the
reason was not the model. On Ollama 0.18.0 with `nomic-embed-text`, every
capitalised token collapses onto a single vector. Measured: `cos("Apple",
"Cat") = 1.0000`, and `cos("Apple", "Zebra") = 1.0000`, while the lower-cased
forms behave normally (`cos("apple", "petrobras") = 0.4075`). Two sentences
differing only in a company name came back byte for byte identical. Across the
training set, 287 distinct texts produced **119 distinct vectors**.

So a third of the hard negatives were literally unrepresentable: the input for
"what was Apple's revenue in 2021?" and "what was Petrobras's revenue in 2021?"
was the same array of numbers, with opposite labels. Lower-casing the text
before encoding is one line, and it moved validation accuracy from 0.933 to
0.980, the unseen subjects from 0.820 to 0.853, `tone` from 0.653 to 0.760 and
`answerable` from 0.600 to 0.680.

Two things follow. A frozen encoder puts a hard ceiling on the head above it:
whatever it does not represent cannot be learned, no matter how the head is
trained. And the way to find that ceiling is to measure the encoder directly
rather than to keep tuning the part that is not at fault.

`answerable` at 0.680 and `tone` at 0.760 are the honest remaining gaps. Telling
2021 from 2019 in a long sentence is a single token in a mean-pooled vector, and
deciding whether a fact is good news needs polarity (more litigation is bad,
fewer infections is good) which does not transfer from four subjects to two new
ones the way semantic alignment does.

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
  tasks.py        the synthetic training tasks, hard negatives included
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
