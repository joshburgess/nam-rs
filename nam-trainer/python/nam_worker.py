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
import importlib.metadata
import os
import re
import sys
import traceback
import inspect
import shutil
import threading
from pathlib import Path

PROTOCOL_VERSION = 3
RUN_ID = "uninitialized"
EVENT_SEQUENCE = 0
CURRENT_FILE_INDEX = None
CANCEL_REQUESTED = threading.Event()
MIN_PACKED_A2_VERSION = (0, 13, 0)

# Force matplotlib to use the non-interactive Agg backend BEFORE anything
# imports it. The worker runs as a headless child process (CREATE_NO_WINDOW
# on Windows), so GUI backends like TkAgg crash when they try to open a
# display. This must happen before pytorch_lightning or nam import
# matplotlib.
os.environ["MPLBACKEND"] = "Agg"


def supports_packed_a2_version(version):
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?",
        version,
    )
    return bool(
        match
        and tuple(int(part) for part in match.groups()) >= MIN_PACKED_A2_VERSION
    )


def emit(event: dict):
    """Write a JSON event to stdout and flush immediately."""
    global EVENT_SEQUENCE
    EVENT_SEQUENCE += 1
    event.setdefault("protocol_version", PROTOCOL_VERSION)
    event.setdefault("run_id", RUN_ID)
    event.setdefault("file_index", CURRENT_FILE_INDEX)
    event.setdefault("sequence", EVENT_SEQUENCE)
    try:
        print(json.dumps(event), flush=True)
    except OSError:
        # stdout pipe may be broken (e.g. after a CUDA crash corrupts
        # process state). Fall back to stderr so the Rust side can still
        # capture the message via the stderr drain thread.
        print(json.dumps(event), file=sys.stderr, flush=True)


def classify_error(error):
    """Map Python failures to stable protocol error categories."""
    message = str(error).lower()
    if (
        "no module named" in message
        or "missing dependency" in message
        or "not installed" in message
    ):
        return "dependency"
    if (
        "data checks failed" in message
        or "sample rate" in message
        or "audio" in message
        or "metadata" in message
    ):
        return "data_validation"
    if (
        "cuda" in message
        or "cudnn" in message
        or "mps" in message
        or "out of memory" in message
    ):
        return "device"
    if "cancel" in message or "keyboardinterrupt" in message:
        return "user_cancel"
    return "training"


def listen_for_commands():
    """Receive control commands without blocking the training thread."""
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            continue
        if command.get("command") == "cancel":
            CANCEL_REQUESTED.set()
            return


def main():
    global RUN_ID
    global CURRENT_FILE_INDEX

    # Read the training request from stdin
    try:
        raw = sys.stdin.readline()
        if not raw.strip():
            emit({
                "type": "error",
                "error_kind": "protocol",
                "message": "No input received on stdin",
            })
            sys.exit(1)
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        emit({
            "type": "error",
            "error_kind": "protocol",
            "message": f"Invalid JSON input: {e}",
        })
        sys.exit(1)

    RUN_ID = request.get("run_id", "missing")
    request_protocol_version = request.get("protocol_version")
    if request_protocol_version != PROTOCOL_VERSION:
        emit({
            "type": "error",
            "error_kind": "protocol",
            "message": (
                f"Worker protocol {PROTOCOL_VERSION} cannot process "
                f"request protocol {request_protocol_version!r}"
            ),
        })
        sys.exit(2)

    threading.Thread(target=listen_for_commands, daemon=True).start()

    # Import NAM after reading stdin so startup errors are caught
    try:
        import pytorch_lightning as pl
        from nam.train import core
        from nam.train import full as nam_full
        from nam.models.metadata import UserMetadata
    except ImportError as e:
        emit({
            "type": "error",
            "error_kind": "dependency",
            "message": (
                f"Missing dependency: {e}. "
                "Install with: pip install --upgrade neural-amp-modeler"
            ),
        })
        sys.exit(1)

    if request.get("packed", False):
        try:
            nam_version = importlib.metadata.version("neural-amp-modeler")
        except importlib.metadata.PackageNotFoundError:
            nam_version = "unknown"
        if not supports_packed_a2_version(nam_version):
            emit({
                "type": "error",
                "error_kind": "dependency",
                "message": (
                    "Packed A2 training requires neural-amp-modeler >= 0.13.0 "
                    f"(installed: {nam_version}). Upgrade with: "
                    "pip install --upgrade 'neural-amp-modeler>=0.13.0'"
                ),
            })
            sys.exit(1)

    # Custom callback for JSON progress reporting. We use
    # on_validation_epoch_end so both training and validation metrics
    # are available. on_train_epoch_end fires before validation runs,
    # so val_loss/ESR would be stale or missing.
    class JsonProgressCallback(pl.Callback):
        """Reports training progress as JSON lines to stdout."""

        def __init__(self):
            self._logged_keys = False

        def on_validation_epoch_end(self, trainer, pl_module):
            if CANCEL_REQUESTED.is_set():
                raise KeyboardInterrupt("Training cancelled by user")
            metrics = trainer.callback_metrics

            # Log available metric keys on the first epoch to aid debugging
            if not self._logged_keys:
                emit({"type": "log", "message": f"Available metrics: {sorted(metrics.keys())}"})
                self._logged_keys = True

            # NAM logs training loss as "loss", validation as "val_loss".
            # ESR may appear as "ESR" or may just be val_loss (NAM uses
            # ESR as the loss function). Try multiple key names.
            train_loss = float(
                metrics.get("loss", metrics.get("train_loss",
                metrics.get("train_loss_epoch", 0.0)))
            )
            val_loss = float(metrics.get("val_loss", 0.0))
            esr = float(
                metrics.get("ESR", metrics.get("val_esr",
                metrics.get("val_loss", 0.0)))
            )

            emit({
                "type": "epoch_end",
                "epoch": trainer.current_epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "esr": esr,
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
    text_metadata_keys = (
        "name",
        "modeled_by",
        "gear_make",
        "gear_model",
        "gear_type",
        "tone_type",
    )
    level_metadata_keys = ("input_level_dbu", "output_level_dbu")
    if any(meta.get(k) for k in text_metadata_keys) or any(
        meta.get(k) is not None for k in level_metadata_keys
    ):
        user_metadata_dict = {}
        for key in text_metadata_keys:
            val = meta.get(key)
            if val:
                user_metadata_dict[key] = val
        for key in level_metadata_keys:
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
    train_signature = inspect.signature(core.train)

    def call_core_train(**kwargs):
        supported = {
            key: value for key, value in kwargs.items()
            if key in train_signature.parameters
        }
        return core.train(**supported)

    def supports_full_config_training():
        required_core_attrs = (
            "_detect_input_version",
            "_analyze_latency",
            "_get_final_latency",
            "_check_data",
            "_get_configs",
        )
        return all(hasattr(core, name) for name in required_core_attrs) and hasattr(
            nam_full, "main"
        )

    def build_packed_full_configs(output_path):
        input_version, _strong_match = core._detect_input_version(input_path)
        latency_analysis = core._analyze_latency(
            request.get("latency"),
            input_version,
            input_path,
            output_path,
            silent=True,
        )
        final_latency = core._get_final_latency(latency_analysis)
        data_check_output = core._check_data(
            input_path,
            output_path,
            input_version,
            final_latency,
            True,
        )
        if (
            data_check_output is not None
            and not data_check_output.passed
            and not request.get("ignore_checks", False)
        ):
            raise RuntimeError(
                "NAM data checks failed. Enable ignore checks to train anyway."
            )

        data_config, model_config, learning_config = core._get_configs(
            input_version,
            input_path,
            output_path,
            final_latency,
            request.get("epochs", 100),
            request.get("num_output_samples_per_datum", 8192),
            request.get("batch_size", 16),
        )

        if "optimizer" in model_config and "lr" in model_config["optimizer"]:
            model_config["optimizer"]["lr"] = request.get("lr", 0.004)

        learning_config.setdefault("trainer", {})
        learning_config["trainer"]["max_epochs"] = request.get("epochs", 100)
        learning_config.setdefault("train_dataloader", {})
        learning_config["train_dataloader"]["batch_size"] = request.get(
            "batch_size", 16
        )

        return data_config, model_config, learning_config

    def train_with_full_config(output_path, train_dir):
        data_config, model_config, learning_config = build_packed_full_configs(
            output_path
        )
        train_dir_path = Path(train_dir)
        train_dir_path.mkdir(parents=True, exist_ok=True)
        nam_full.main(
            data_config,
            model_config,
            learning_config,
            train_dir_path,
            no_show=True,
            make_plots=request.get("save_plot", True),
        )
        model_path = train_dir_path / "model.nam"
        if not model_path.exists():
            raise RuntimeError(f"Packed full training did not export {model_path}")
        return str(model_path)

    def sanitize_model_basename(value):
        sanitized = "".join(
            "_" if ch in '/\\:*?"<>|' or ord(ch) < 32 else ch
            for ch in str(value).strip()
        ).strip(". ")
        return sanitized or "model"

    def output_stem(output_path):
        return os.path.splitext(os.path.basename(output_path))[0] or "model"

    def model_basename(index, output_path):
        stem = output_stem(output_path)
        custom_name = request.get("output_model_basename")
        if len(output_paths) == 1 and custom_name and str(custom_name).strip():
            return sanitize_model_basename(custom_name)
        template = request.get("batch_name_template") or "{stem}"
        rendered = str(template).replace("{stem}", stem).replace("{index}", str(index + 1))
        return sanitize_model_basename(rendered)

    for index, output_path in enumerate(output_paths):
        if CANCEL_REQUESTED.is_set():
            break
        CURRENT_FILE_INDEX = index
        basename = model_basename(index, output_path)

        emit({
            "type": "training_start",
            "file": output_path,
            "total_epochs": request.get("epochs", 100),
        })

        try:
            architecture = request.get("architecture", "standard")
            if request.get("packed", False):
                architecture = "packed"
            train_dir = os.path.join(destination, basename)
            if (
                request.get("packed", False)
                and request.get("use_full_config_trainer", False)
                and supports_full_config_training()
            ):
                emit({
                    "type": "log",
                    "message": "Using upstream packed full-config trainer path",
                })
                exported_model = train_with_full_config(output_path, train_dir)
            else:
                trained_model = call_core_train(
                    input_path=input_path,
                    output_path=output_path,
                    train_path=train_dir,
                    epochs=request.get("epochs", 100),
                    latency=request.get("latency"),
                    architecture=architecture,
                    batch_size=request.get("batch_size", 16),
                    ny=request.get("num_output_samples_per_datum", 8192),
                    lr=request.get("lr", 0.004),
                    lr_decay=request.get("lr_decay", 0.007),
                    seed=0,
                    save_plot=request.get("save_plot", True),
                    silent=True,  # No matplotlib popups
                    modelname=basename,
                    ignore_checks=request.get("ignore_checks", False),
                    fit_mrstft=request.get("fit_mrstft", True),
                    threshold_esr=request.get("threshold_esr"),
                    user_metadata=user_metadata,
                )
                exported_model = None

            # Find the .nam file. core.train() puts it deep inside
            # lightning_logs/version_N/checkpoints/. Search for it and
            # copy to the output directory with a clean name.
            found_nam = exported_model
            if found_nam is None:
                for root, dirs, files in os.walk(train_dir):
                    for f in files:
                        if f.endswith(".nam"):
                            found_nam = os.path.join(root, f)
                            break
                    if found_nam:
                        break

            if found_nam:
                final_path = os.path.join(destination, f"{basename}.nam")
                shutil.copy2(found_nam, final_path)
                model_path = final_path
            else:
                model_path = os.path.join(train_dir, f"{basename}.nam")

            emit({
                "type": "training_complete",
                "file": output_path,
                "validation_esr": 0.0,
                "model_path": model_path,
            })

        except BaseException as e:
            emit({"type": "log", "message": traceback.format_exc()})
            emit({
                "type": "training_failed",
                "file": output_path,
                "error_kind": classify_error(e),
                "error": str(e),
            })
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                break

    CURRENT_FILE_INDEX = None
    emit({"type": "all_complete"})


if __name__ == "__main__":
    main()
    # Replace stdout/stderr with devnull before interpreter shutdown so
    # Python's atexit flush doesn't fail on the broken pipe and print
    # "Exception ignored on flushing sys.stdout: OSError".
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    except OSError:
        pass
