# Install a nam-rs release

nam-rs releases are not signed with an Apple Developer ID or Windows
Authenticode certificate, and they are not notarized. Download artifacts only
from the [nam-rs releases page](https://github.com/joshburgess/nam-rs/releases).
Verify the checksum and GitHub attestation before installing them.

## Verify the download

Download the platform archive, `SHA256SUMS`, and `nam-rs.spdx.json` from the
same release. From the download directory, run:

```sh
shasum -a 256 --check SHA256SUMS
gh attestation verify nam-rs-v0.2.0-PLATFORM.zip \
  --repo joshburgess/nam-rs
```

On Windows PowerShell, compare the archive with its line in `SHA256SUMS`:

```powershell
Get-FileHash .\nam-rs-v0.2.0-windows-x86_64.zip -Algorithm SHA256
gh attestation verify .\nam-rs-v0.2.0-windows-x86_64.zip `
  --repo joshburgess/nam-rs
```

Replace `PLATFORM` with the archive name for your operating system. A valid
attestation identifies `joshburgess/nam-rs` and the release workflow that
built the archive.

## macOS

The macOS archive targets Apple Silicon. Extract it, then copy:

- `plugins/nam-plugin.vst3` to `~/Library/Audio/Plug-Ins/VST3/`
- `plugins/nam-plugin.clap` to `~/Library/Audio/Plug-Ins/CLAP/`
- `bin/nam-trainer` to `~/Applications/`

The macOS files do not carry an Apple Developer ID signature and are not
notarized. Plugin bundles may contain an ad-hoc signature used for bundle
integrity. An ad-hoc signature does not identify or authenticate the
publisher. macOS may block the trainer the first time you open it. In Finder,
Control-click `nam-trainer`, choose **Open**, then confirm **Open**. Do not
remove quarantine attributes from files whose checksum and attestation you
have not verified.

Remove the three copied files to uninstall the release.

## Windows

Extract the Windows archive, then copy:

- `plugins\nam-plugin.vst3` to
  `%LOCALAPPDATA%\Programs\Common\VST3\`
- `plugins\nam-plugin.clap` to
  `%LOCALAPPDATA%\Programs\Common\CLAP\`
- `bin\nam-trainer.exe` to `%LOCALAPPDATA%\Programs\nam-rs\`

Windows SmartScreen may warn that `nam-trainer.exe` has an unknown publisher.
Confirm the SHA-256 digest and GitHub attestation before choosing to run it.
The release does not claim an Authenticode signature.

Remove the three copied files to uninstall the release.

## Linux

Extract the Linux archive, then copy:

- `plugins/nam-plugin.vst3` to `~/.vst3/`
- `plugins/nam-plugin.clap` to `~/.clap/`
- `bin/nam-trainer` to `~/.local/bin/`

Make the trainer executable if your archive tool did not retain its mode:

```sh
chmod 755 ~/.local/bin/nam-trainer
```

The Linux build requires glibc 2.35 or newer. Remove the three copied files to
uninstall the release.

## Upgrade

Close hosts using NAM and exit the trainer. Verify the new archive, then
replace all three files from the previous release. Keep the VST3 bundle
intact. Copying only its shared library leaves a mixed-version installation.
