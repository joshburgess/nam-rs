# nam-rs v0.2.0

This release adds current post-A2 NeuralAmpModelerCore compatibility, packed
A2 training and plugin controls, allocation-free plugin processing, optimized
inference, and validated VST3 and CLAP bundles for macOS, Windows, and Linux.

The release workflow installs each archive into the platform's user plugin
directories, validates the installed VST3 and CLAP plugins, launches the
installed trainer, tests replacement and uninstallation, and publishes only
after those checks pass.

## Verify before installing

These files are not signed with an Apple Developer ID or Windows Authenticode
certificate, and they are not notarized. The release includes `SHA256SUMS`, an
SPDX software bill of materials, and GitHub artifact attestations. Verify the
archive before accepting an operating-system warning or copying plugins into
a host search path.

See the [installation and verification guide](https://github.com/joshburgess/nam-rs/blob/v0.2.0/docs/installing-releases.md)
for platform paths, verification commands, publisher-warning guidance,
upgrades, and uninstallation.
