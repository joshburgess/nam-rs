# NAM Trainer GUI Improvements

## Highest Value

- [x] Add worker integration tests. Test `nam_worker.py` with mocked `nam.train.core` and `nam.train.full` modules so request routing, errors, metadata, `packed`, `ny`, `ignore_checks`, and full-config mode are verified without requiring PyTorch.
- [x] Stop writing the worker script to a fixed temp path: `std::env::temp_dir().join("nam_worker.py")`. Two app instances can collide. Use a unique per-process temp path or `tempfile`.
- [x] Replace `serde_json::to_string(&request).unwrap_or_default()` with explicit error handling. If request serialization fails, report a Rust-side GUI error instead of sending empty or invalid JSON to Python.
- [x] Improve cancellation cleanup. `kill()` is called, but the cancel path does not always wait/reap immediately. Prefer a clear cancel protocol or a kill-then-wait path.

## Training Robustness

- [x] Add a preflight environment report before training: Python version, NAM version, torch version, CUDA/MPS availability, selected device, and whether packed/full-config APIs are present.
- [x] Validate metadata fields in the GUI before training, especially enum values and dBu numeric inputs, instead of relying on Python warnings.
- [x] Validate output destination writability and available disk space before launching training.
- [x] Make full-config trainer limitations explicit in the UI. It currently skips GUI metadata export support, so warn or disable that mode when metadata fields are filled.

## UX And Reliability

- [x] Show the exact generated request/config summary before training in an expandable panel.
- [x] Add “Open output folder” and “Reveal model” actions after completion.
- [x] Save persistent training logs beside the model.
- [x] Add clearer batch progress details per output file, not only total epochs.
- [x] Improve error classification: dependency error, data validation failure, CUDA failure, subprocess crash, and user cancel.

## Performance

- [x] Avoid cloning or retaining large logs indefinitely. Cap `training_log` or stream to a file with a bounded in-memory tail.
- [x] Cache Python discovery results and refresh only on demand. Python discovery currently shells out and scans conda environments.
- [x] Avoid excessive UI repainting/log processing during long training when stderr is noisy.

## Test Coverage

- [x] Add unit tests for `validate_audio_files`.
- [x] Add tests for settings persistence of the new trainer options.
- [x] Add tests for `event_to_message`.
- [x] Add worker subprocess tests using a fake Python script that emits stdout/stderr protocol events.
- [x] Add a golden JSON test for the Rust `TrainRequest` schema so protocol changes are intentional.

## Next-Tier Reliability

- [x] Add real GUI flow tests around app state transitions: configure a run, start fake training, cancel, complete, and inspect the resulting state.
- [x] Stream training logs to a persistent file during training while keeping only a bounded in-memory tail.
- [x] Save a training run manifest beside each model with the generated request, Python executable, NAM version, Torch version, selected device, input and output paths, and timestamp.
- [x] Add output overwrite protection for model, log, and manifest files before training starts.
- [x] Estimate required destination disk space from input/output audio sizes and batch count instead of only using a fixed free-space floor.
- [x] Add a copyable diagnostics summary covering Python path, package versions, detected devices, selected config, and recent errors.
- [x] Persist enough active-run state that a restarted app can detect an incomplete training run and point to its logs/output directory.
- [x] Add a device fallback policy for unavailable CUDA/MPS selections before training starts.
- [x] Add golden worker protocol fixtures for request and emitted event JSON.
- [x] Add UI regression smoke tests for default, error, running, cancelled, and completed states.

## Polish And Maintainability

- [x] Move trainer GUI core logic toward smaller modules so `app.rs` is not the only home for artifacts, diagnostics, and error classification.
- [x] Introduce a typed training-run artifact model for predicted model paths, log paths, manifest paths, and overwrite validation.
- [x] Add a real fake-worker mode that exercises the same worker protocol used by Python training.
- [x] Make overwrite handling user-controllable by allowing an explicit overwrite setting while keeping protection on by default.
- [x] Add structured error details with an error kind separate from the raw source message.
- [x] Persist recent successful run history with model path, manifest path, ESR, architecture, device, and timestamp.
- [x] Add app-level snapshot-style regression tests for important rendered state text and controls.
- [x] Add focused CI checks for the trainer GUI, including Rust checks and Python syntax checks.
- [x] Improve manifest completeness with app version, OS, architecture, package build metadata, and WAV metadata.
- [x] Add cancellation manifest status so cancelled runs are recorded as cancelled instead of only clearing recovery state.

## Library Split And Product Polish

- [x] Split `nam-trainer` into a real library plus binary with `main.rs` only launching the app.
- [x] Add a proper Run History UI with model path, ESR, architecture, device, timestamp, reveal/open actions, and diagnostics copy.
- [x] Make overwrite handling more ergonomic by showing exact conflicting files and offering overwrite, rename, or choose-folder paths.
- [x] Add a dedicated diagnostics panel for environment, request, logs, active run paths, and package versions.
- [x] Move error classification into its own module with focused tests.
- [x] Add richer worker protocol versioning to Rust requests and Python worker events.
- [x] Add UI tests around clicked control behavior for overwrite, diagnostics, cancel, reset, and train-again paths.
- [x] Add model/output naming controls instead of only deriving model names from output WAV filenames.

## Reliability And Safety Follow-Up

- [x] Separate per-file completion and failure events from final batch completion, including mixed-success and multi-file tests.
- [x] Reject generated model, log, and manifest name collisions even when overwriting is enabled.
- [x] Generate collision-resistant run IDs with timestamp, process, and atomic counter components.
- [x] Save settings and manifests atomically, propagate persistence errors, and version the settings schema.
- [x] Use remembered cancellation and a bounded worker channel that drops only lossy progress messages under load.
- [x] Require matching protocol versions and return structured worker error kinds.
- [x] Protect destructive environment removal with confirmation and a managed-install marker.
- [x] Pin Miniforge installer versions and verify SHA-256 checksums before execution.
- [x] Separate blocking validation errors from non-blocking warnings.
- [x] Add a readiness summary and collapse secondary history and diagnostics content.
- [x] Group run status, worker ownership, artifacts, progress, timing, and results in a typed run context.
- [x] Narrow the library facade to intentional public exports.

## Lifecycle, Filesystem, And Supply-Chain Hardening

- [x] Encode idle, running, finishing, and finished run states as lifecycle variants that carry only their valid resources.
- [x] Give the worker-owner thread exclusive subprocess ownership and route cancellation through a command channel.
- [x] Resolve filesystem aliases when checking artifact collisions, stage each run separately, and atomically promote completed models.
- [x] Fault-inject atomic persistence at write, sync, replace, and directory-sync boundaries, including rollback, cleanup, and permission tests.
- [x] Surface save, open, and reveal failures as dismissible user-facing errors.
- [x] Extract environment management, diagnostics, and run-manifest ownership from `app.rs`.
- [x] Add property and adversarial tests for artifact names, settings migration, protocol input, malformed worker output, reservations, and cancellation timing.
- [x] Enforce dependency advisories, licenses, bans, and sources in CI, pin GitHub Actions and git dependencies, and update known vulnerable dependencies.

## Suggested First Step

Start with the worker integration test harness and the unique temp worker path. Those changes would make future trainer GUI work safer and reduce the risk of subprocess lifecycle bugs.
