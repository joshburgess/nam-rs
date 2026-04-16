#!/usr/bin/env python3
"""NAM training worker process.

Reads a single JSON training request from stdin, runs training via
nam.train.core, and writes JSON progress events to stdout.

Protocol:
  stdin:  single JSON line with TrainRequest
  stdout: one JSON line per event (epoch_end, training_complete, error, etc.)

Progress is captured by monkey-patching PyTorch Lightning's Trainer to
inject a custom callback, since core.train() doesn't expose a callback parameter.
"""

import json
import os
import sys
import traceback


def emit(event: dict):
    """Write a JSON event to stdout and flush immediately."""
    try:
        print(json.dumps(event), flush=True)
    except OSError:
        # stdout pipe may be broken (e.g. after a CUDA crash corrupts
        # process state). Fall back to stderr so the Rust side can still
        # capture the message via the stderr drain thread.
        print(json.dumps(event), file=sys.stderr, flush=True)


def main():
    # Read the training request from stdin
    try:
        raw = sys.stdin.readline()
        if not raw.strip():
            emit({"type": "error", "message": "No input received on stdin"})
            sys.exit(1)
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        emit({"type": "error", "message": f"Invalid JSON input: {e}"})
        sys.exit(1)

    # Import NAM after reading stdin so startup errors are caught
    try:
        import pytorch_lightning as pl
        from nam.train import core
        from nam.models.metadata import UserMetadata
    except ImportError as e:
        emit({"type": "error", "message": f"Missing dependency: {e}. "
              "Install with: pip install neural-amp-modeler"})
        sys.exit(1)

    # Custom callback for JSON progress reporting
    class JsonProgressCallback(pl.Callback):
        """Reports training progress as JSON lines to stdout."""

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            emit({
                "type": "epoch_end",
                "epoch": trainer.current_epoch + 1,
                "train_loss": float(metrics.get("train_loss", 0.0)),
                "val_loss": float(metrics.get("val_loss", 0.0)),
                "esr": float(metrics.get("ESR", metrics.get("val_loss", 0.0))),
            })

    # Monkey-patch the Trainer to inject our callback
    _original_trainer_init = pl.Trainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        _original_trainer_init(self, *args, **kwargs)
        self.callbacks.append(JsonProgressCallback())

    pl.Trainer.__init__ = _patched_trainer_init

    # Set device via environment variable if specified
    device = request.get("device", "")
    if device.startswith("cuda:"):
        gpu_idx = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    elif device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Build user metadata
    meta = request.get("metadata", {})
    user_metadata = None
    if any(meta.get(k) for k in ("name", "modeled_by", "gear_make", "gear_model",
                                   "gear_type", "tone_type")):
        user_metadata_dict = {}
        for key in ("name", "modeled_by", "gear_make", "gear_model",
                     "gear_type", "tone_type"):
            val = meta.get(key)
            if val:
                user_metadata_dict[key] = val
        for key in ("input_level_dbu", "output_level_dbu"):
            val = meta.get(key)
            if val is not None:
                user_metadata_dict[key] = val
        try:
            user_metadata = UserMetadata(**user_metadata_dict)
        except Exception as e:
            emit({"type": "log", "message": f"Warning: invalid metadata: {e}"})

    input_path = request["input_path"]
    output_paths = request["output_paths"]
    destination = request["destination"]

    for output_path in output_paths:
        basename = os.path.splitext(os.path.basename(output_path))[0]

        emit({
            "type": "training_start",
            "file": output_path,
            "total_epochs": request.get("epochs", 100),
        })

        try:
            trained_model = core.train(
                input_path=input_path,
                output_path=output_path,
                train_path=os.path.join(destination, basename),
                epochs=request.get("epochs", 100),
                latency=request.get("latency"),
                architecture=request.get("architecture", "standard"),
                batch_size=request.get("batch_size", 16),
                lr=request.get("lr", 0.004),
                lr_decay=request.get("lr_decay", 0.007),
                seed=0,
                save_plot=request.get("save_plot", True),
                silent=True,  # No matplotlib popups
                modelname=basename,
                ignore_checks=request.get("ignore_checks", True),
                fit_mrstft=request.get("fit_mrstft", True),
                threshold_esr=request.get("threshold_esr"),
                user_metadata=user_metadata,
            )

            # Find the exported model path
            model_path = os.path.join(destination, basename, f"{basename}.nam")
            if not os.path.exists(model_path):
                for root, dirs, files in os.walk(os.path.join(destination, basename)):
                    for f in files:
                        if f.endswith(".nam"):
                            model_path = os.path.join(root, f)
                            break

            emit({
                "type": "training_complete",
                "file": output_path,
                "validation_esr": 0.0,
                "model_path": model_path,
            })

        except BaseException as e:
            emit({
                "type": "training_failed",
                "file": output_path,
                "error": str(e),
            })
            emit({"type": "log", "message": traceback.format_exc()})
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                break

    emit({"type": "all_complete"})


if __name__ == "__main__":
    main()
