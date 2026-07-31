# Local patches

This source is based on [NIH-plug commit `28b149ec4d62757d0b448809148a0c3ca6e09a95`](https://github.com/robbert-vdh/nih-plug/tree/28b149ec4d62757d0b448809148a0c3ca6e09a95). Only the root crate, derive crate, and egui integration needed by nam-rs are vendored. The upstream workspace members are omitted.

The wrappers include these local compatibility fixes:

- Reject state payloads above 64 MiB and use fallible reservation before reading the host-provided length.
- Request a host parameter-value rescan after state restoration.
- Backport maintained-fork commit [`83d934c`](https://codeberg.org/RustAudio/nice-plug/commit/83d934cd0a538f402c1a80bac71ca28039928d7c), using weak editor task-executor captures so CLAP and VST3 wrappers can be destroyed without reference cycles.

The state changes are exercised by CLAP validator 0.4.1. Wrapper destruction is exercised by both plugin validators in the release workflow.

The vendored subset also includes current rustfmt output and narrow warning accommodations needed by the nam-rs warnings-as-errors build.
