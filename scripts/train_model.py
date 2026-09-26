"""Train the scorer end to end and print what it measured.

The test split deliberately uses only the domains the training split never
touched. Reporting accuracy on phrasings of familiar facts would measure
memorisation; the number worth publishing is the one from subjects the weights
have not seen.
"""

import argparse
import sys
from pathlib import Path

from brier.calibration import assess, calibrate, coverage_curve
from brier.encoder import CachedEncoder, EncoderError, OllamaEncoder
from brier.model import save
from brier.tasks import DOMAINS, generate
from brier.train import TrainingConfig, by_name, featurise, fit

HELD_OUT = frozenset({"infections", "attendance"})
TRAIN_SIZE = 900
VALIDATION_SIZE = 300
TEST_SIZE = 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/brier.npz"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/encoder"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    seen = frozenset(DOMAINS) - HELD_OUT
    training = generate(TRAIN_SIZE, seed=arguments.seed, held_out=HELD_OUT)
    validation = generate(VALIDATION_SIZE, seed=arguments.seed + 1, held_out=HELD_OUT)
    test = generate(TEST_SIZE, seed=arguments.seed + 2, held_out=seen)

    print(f"train/val domains: {sorted(seen)}")
    print(f"test domains     : {sorted(HELD_OUT)}  (never seen in training)")

    try:
        encoder = CachedEncoder(OllamaEncoder(), arguments.cache)
        dimension = encoder.dimension
        print(f"encoder dimension: {dimension} (frozen)")
        batches = {
            name: featurise(split, encoder)
            for name, split in (("train", training), ("val", validation), ("test", test))
        }
    except EncoderError as error:
        print(f"encoder unavailable: {error}", file=sys.stderr)
        print("start ollama and pull nomic-embed-text, then run again", file=sys.stderr)
        return 1

    parameters, report = fit(
        batches["train"],
        batches["val"],
        dimension,
        TrainingConfig(epochs=arguments.epochs, seed=arguments.seed),
    )
    print(f"\ntrained {report.parameter_count:,} parameters, best epoch {report.best_epoch}")
    print(
        f"  val loss {report.best.holdout_loss:.4f}  "
        f"val accuracy {report.best.holdout_accuracy:.3f}"
    )

    calibrated, on_validation = calibrate(parameters, batches["val"])
    print(f"\ntemperature {calibrated.temperature} fitted on validation")
    _show("validation", on_validation)

    on_test = assess(calibrated, batches["test"])
    _show("test (unseen domains), temperature from validation", on_test)

    # A temperature fitted in one domain is not a property of the model, it is a
    # property of the pair. Splitting the unseen domains in half and refitting on
    # the first half measures how much of the gap is the temperature and how much
    # is the model genuinely not knowing the new subject.
    half = len(batches["test"]) // 2
    recalibrated, _ = calibrate(parameters, batches["test"][:half])
    on_rest = assess(recalibrated, batches["test"][half:])
    print(f"\nrefitting temperature on half the unseen domains gives {recalibrated.temperature}")
    _show("test (unseen domains), temperature refitted there", on_rest)

    print("\naccuracy per question family on unseen domains:")
    for name, (count, accuracy) in by_name(recalibrated, batches["test"]).items():
        print(f"  {name:<10} n={count:<4} accuracy {accuracy:.3f}")

    print("\nabstention on unseen domains (temperature refitted there):")
    for point in coverage_curve(recalibrated, batches["test"][half:], (0.0, 0.6, 0.7, 0.8, 0.9)):
        print(
            f"  threshold {point.threshold:.1f}  answers {point.coverage:6.1%} of questions  "
            f"accuracy there {point.accuracy_when_answered:.3f}"
        )

    save(calibrated, arguments.output)
    print(f"\nweights written to {arguments.output}")
    return 0


def _show(label: str, report) -> None:
    print(f"\n{label}:")
    print(f"  accuracy   {report.accuracy:.3f}")
    print(
        f"  confidence {report.mean_confidence:.3f}  (overconfidence {report.overconfidence:+.3f})"
    )
    print(f"  ECE        {report.expected_calibration_error:.4f}")
    print(f"  Brier      {report.brier:.4f}")
    print("  reliability:")
    for slice_ in report.bins:
        print(
            f"    {slice_.lower:.2f}-{slice_.upper:.2f}  n={slice_.count:<4} "
            f"confidence {slice_.mean_confidence:.3f}  accuracy {slice_.accuracy:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
