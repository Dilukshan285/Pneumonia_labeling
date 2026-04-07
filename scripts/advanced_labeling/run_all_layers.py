"""
Master Runner — Advanced Multi-System Adversarial Consensus Pipeline

Runs all 7 layers in order:
  Layer 1: Snorkel Ultra-Strict Seeds     (~2 min, CPU)
  Layer 4: Assertion Classification       (~15 min, CPU)
  Layer 2: NLI Ensemble (DeBERTa)        (~5-7 hours, GPU)
  Layer 3: PubMedBERT Fine-tune + Predict (~50 min, GPU)
  Layers 5-7: Consensus + Adversarial    (~2 min, CPU)

Layers 1 and 4 run first (CPU only, fast).
Then Layer 2 (GPU, longest).
Then Layer 3 (GPU, needs Layer 1 seeds).
Finally Layers 5-7 (CPU, combines everything).

Usage:
  python run_all_layers.py           # Run everything
  python run_all_layers.py --skip 2  # Skip Layer 2 if already done
  python run_all_layers.py --only 1  # Run only Layer 1
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import layer runners
from advanced_labeling.layer1_strict_seeds import main as run_layer1
from advanced_labeling.layer4_assertion import main as run_layer4
from advanced_labeling.layer2_nli_ensemble import main as run_layer2
from advanced_labeling.layer3_pubmedbert import main as run_layer3
from advanced_labeling.layer5_to_7_consensus import main as run_layer5_7


LAYERS = {
    1: ("Layer 1: Snorkel Ultra-Strict Seeds", run_layer1, "~2 min, CPU"),
    4: ("Layer 4: Assertion Classification", run_layer4, "~15 min, CPU"),
    2: ("Layer 2: NLI Ensemble (DeBERTa)", run_layer2, "~5-7 hrs, GPU"),
    3: ("Layer 3: PubMedBERT Fine-tune", run_layer3, "~50 min, GPU"),
    57: ("Layers 5-7: Consensus + Adversarial + Calibration", run_layer5_7, "~2 min, CPU"),
}

# Execution order (Layer 1 and 4 first as they're fast CPU tasks)
EXEC_ORDER = [1, 4, 2, 3, 57]


def main():
    parser = argparse.ArgumentParser(description="Advanced Consensus Pipeline Runner")
    parser.add_argument("--skip", type=int, nargs="+", default=[],
                        help="Layer numbers to skip (e.g., --skip 2 3)")
    parser.add_argument("--only", type=int, nargs="+", default=[],
                        help="Only run these layers (e.g., --only 1 4)")
    args = parser.parse_args()

    print("=" * 70)
    print("ADVANCED MULTI-SYSTEM ADVERSARIAL CONSENSUS PIPELINE")
    print("=" * 70)
    print()
    print("  Execution plan:")
    for layer_id in EXEC_ORDER:
        name, _, runtime = LAYERS[layer_id]
        skip = layer_id in args.skip
        only = args.only and layer_id not in args.only
        status = " [SKIP]" if (skip or only) else ""
        print(f"    {name} ({runtime}){status}")
    print()

    t_total_start = time.time()
    results = {}

    for layer_id in EXEC_ORDER:
        name, runner, runtime = LAYERS[layer_id]

        if layer_id in args.skip:
            print(f"  ⏭ Skipping {name}")
            continue
        if args.only and layer_id not in args.only:
            print(f"  ⏭ Skipping {name} (--only filter)")
            continue

        print()
        print(f"  ▶ Starting {name} ({runtime})")
        print()

        t_start = time.time()
        try:
            exit_code = runner()
            elapsed = time.time() - t_start

            if exit_code == 0:
                results[layer_id] = ("✓ SUCCESS", elapsed)
                print(f"\n  ✓ {name} completed in {elapsed:.1f}s\n")
            else:
                results[layer_id] = (f"✗ FAILED (exit={exit_code})", elapsed)
                print(f"\n  ✗ {name} FAILED with exit code {exit_code}\n")
                print("  Stopping pipeline. Fix the error and re-run with --skip for completed layers.")
                break

        except Exception as e:
            elapsed = time.time() - t_start
            results[layer_id] = (f"✗ ERROR: {str(e)[:50]}", elapsed)
            print(f"\n  ✗ {name} ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            print("\n  Stopping pipeline. Fix the error and re-run with --skip for completed layers.")
            break

    # Summary
    t_total = time.time() - t_total_start
    print()
    print("=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print()
    for layer_id in EXEC_ORDER:
        name = LAYERS[layer_id][0]
        if layer_id in results:
            status, elapsed = results[layer_id]
            print(f"  {status:>20s}  {name}  ({elapsed:.1f}s)")
        else:
            print(f"  {'⏭ SKIPPED':>20s}  {name}")
    print()
    print(f"  Total runtime: {t_total/3600:.1f} hours ({t_total:.0f}s)")
    print("=" * 70)

    # Return 0 only if all completed layers succeeded
    all_ok = all(s.startswith("✓") for s, _ in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
