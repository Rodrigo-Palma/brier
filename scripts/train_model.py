"""Train the scorer, calibrate it out of domain, and report with intervals.

Three splits over disjoint subjects, because two is not enough to tell two
things apart:

- train on four subjects;
- fit the temperature on two subjects the weights never saw, so a calibration
  that fails on new data cannot be blamed on the fit having been done in-domain;
- test on the last two, never seen by either.

Every rate is printed with a Wilson interval, beside what the frozen encoder
answers for free. A number without both of those cannot support a claim.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from brier.baseline import cosine_batches, fit_thresholds
from brier.calibration import assess, calibrate, coverage_curve
from brier.data import identity
from brier.encoder import CachedEncoder, EncoderError, OllamaEncoder
from brier.evaluate import Evaluation, evaluate
from brier.features import FEATURE_BLOCKS
from brier.model import Provenance, save
from brier.statistics import wilson
from brier.tasks import DOMAINS, generate
from brier.train import TrainingConfig, by_tag, featurise, fit

TRAIN_SUBJECTS = frozenset({"revenue", "headcount", "litigation", "rainfall"})
CALIBRATE_SUBJECTS = frozenset({"emissions", "downtime"})
TEST_SUBJECTS = frozenset({"attendance", "infections"})

# Sized to the generated space, not to taste. Probed: four subjects support
# about 1000 distinct examples and two support about 400, with `tone` the
# narrowest family. Asking for more than that used to return duplicates in
# silence; `generate` now refuses, which is how these numbers were found.
TRAIN_SIZE = 800
VALIDATION_SIZE = 200
CALIBRATE_SIZE = 200
TEST_SIZE = 200
THRESHOLDS = (0.0, 0.6, 0.7, 0.8, 0.9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/brier.npz"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/encoder"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    splits = _build_splits(arguments.seed)
    _describe(splits)

    try:
        encoder = CachedEncoder(OllamaEncoder(), arguments.cache)
        dimension = encoder.dimension
        print(f"\nencoder: {dimension} dimensions, frozen, text lower-cased before sending")
        batches = {name: featurise(rows, encoder) for name, rows in splits.items()}
        cosines = {name: cosine_batches(rows, encoder) for name, rows in splits.items()}
    except EncoderError as error:
        print(f"encoder unavailable: {error}", file=sys.stderr)
        print("start ollama and pull nomic-embed-text, then run again", file=sys.stderr)
        return 1

    parameters, report = fit(
        batches["train"],
        batches["validation"],
        dimension,
        TrainingConfig(epochs=arguments.epochs, seed=arguments.seed),
    )
    print(f"\ntrained {report.parameter_count:,} parameters, best epoch {report.best_epoch}")
    print(
        f"  validation loss {report.best.holdout_loss:.4f}  "
        f"accuracy {wilson(round(report.best.holdout_accuracy * VALIDATION_SIZE), VALIDATION_SIZE)}"
    )

    thresholds = fit_thresholds(cosines["train"])
    print(f"\nbaseline thresholds fitted on the training split: {thresholds}")

    calibrated, on_calibration = calibrate(parameters, batches["calibrate"])
    print(f"temperature {calibrated.temperature} fitted on {sorted(CALIBRATE_SUBJECTS)}")

    for name, label in (
        ("calibrate", "calibration subjects (unseen by the weights)"),
        ("test", "test subjects (unseen by anything)"),
    ):
        measured = evaluate(
            calibrated,
            batches[name],
            cosines[name],
            thresholds,
            on_calibration if name == "calibrate" else assess(calibrated, batches[name]),
            distinct=len({identity(row) for row in splits[name]}),
            seed=arguments.seed,
        )
        _show(label, measured)

    print("\nanswerable, one level below the family average:")
    for tag, (count, accuracy) in by_tag(calibrated, batches["test"], "answerable").items():
        print(f"  {tag:<9} {wilson(round(accuracy * count), count)}")

    print("\nabstention on the test subjects:")
    curve = coverage_curve(calibrated, batches["test"], THRESHOLDS)
    for point in curve:
        if not point.answered:
            print(f"  threshold {point.threshold:.1f}  answers nothing")
            continue
        interval = wilson(round(point.accuracy_when_answered * point.answered), point.answered)
        print(
            f"  threshold {point.threshold:.1f}  answers {point.coverage:6.1%}  "
            f"accuracy there {interval}"
        )

    stamped = replace(
        calibrated,
        provenance=Provenance(
            encoder=encoder.fingerprint,
            dimension=dimension,
            feature_blocks=FEATURE_BLOCKS,
            seed=arguments.seed,
        ),
    )
    save(stamped, arguments.output)
    print(f"\nweights written to {arguments.output} ({stamped.provenance.describe()})")
    return 0


def _build_splits(seed: int) -> dict[str, tuple]:
    """Four splits that share no row, over subjects that do not overlap."""
    everything = frozenset(DOMAINS)
    train = generate(TRAIN_SIZE, seed=seed, held_out=everything - TRAIN_SUBJECTS)
    used = frozenset(identity(row) for row in train)
    validation = generate(
        VALIDATION_SIZE, seed=seed + 1, held_out=everything - TRAIN_SUBJECTS, exclude=used
    )
    return {
        "train": train,
        "validation": validation,
        "calibrate": generate(
            CALIBRATE_SIZE, seed=seed + 2, held_out=everything - CALIBRATE_SUBJECTS
        ),
        "test": generate(TEST_SIZE, seed=seed + 3, held_out=everything - TEST_SUBJECTS),
    }


def _describe(splits: dict[str, tuple]) -> None:
    print("subjects per split, with no subject in more than one:")
    print(f"  train + validation : {sorted(TRAIN_SUBJECTS)}")
    print(f"  calibration        : {sorted(CALIBRATE_SUBJECTS)}")
    print(f"  test               : {sorted(TEST_SUBJECTS)}")
    print("\nrows, and how many of them are distinct questions:")
    for name, rows in splits.items():
        print(f"  {name:<11} {len(rows):>5} rows, {len({identity(r) for r in rows}):>5} distinct")

    keys = {name: {identity(r) for r in rows} for name, rows in splits.items()}
    overlaps = {f"{a} n {b}": len(keys[a] & keys[b]) for a in keys for b in keys if a < b}
    print(f"  overlap between splits: {sorted(set(overlaps.values()))}")


def _floor_verdict(calibration) -> str:
    """Whether the measured error says anything at all at this sample size."""
    if calibration.beats_the_floor:
        return "above the floor"
    return "INSIDE the floor: no evidence of miscalibration"


def _show(label: str, measured: Evaluation) -> None:
    calibration = measured.calibration
    print(f"\n{label}: {measured.rows} rows, {measured.distinct} distinct")
    print(f"  model           {measured.model}")
    print(f"  cosine baseline {measured.cosine}")
    print(
        f"  gain over cosine {measured.model.rate - measured.cosine.rate:+.3f} "
        f"[{measured.gain_low:+.3f}, {measured.gain_high:+.3f}]  <- {measured.verdict}"
    )
    print("\n  per family:            model                        cosine        majority")
    for family in measured.families:
        print(
            f"    {family.name:<11} {family.model}   "
            f"{family.cosine.rate:.3f}   {family.majority:.3f}   ({family.gain:+.3f})"
        )

    ranking = measured.ranking
    print("\n  does the confidence know which answers are right?")
    print(
        f"    AUROC {ranking.auroc:.3f} [{ranking.low:.3f}, {ranking.high:.3f}]  "
        f"permutation p={ranking.p_value:.4f}"
    )
    print(
        f"    area under the risk-coverage curve {ranking.risk_coverage_area:.4f} (lower is better)"
    )

    print("\n  calibration:")
    print(f"    ECE          {calibration.expected_calibration_error:.4f}")
    print(
        f"    noise floor  {calibration.noise_floor.rate:.4f} "
        f"[{calibration.noise_floor.low:.4f}, {calibration.noise_floor.high:.4f}]  "
        f"{_floor_verdict(calibration)}"
    )
    print(f"    Brier        {calibration.brier:.4f}")
    print(f"    confidence   {calibration.mean_confidence:.3f} ({calibration.overconfidence:+.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
