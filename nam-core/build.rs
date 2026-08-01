#![allow(clippy::print_stdout)]

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/fast_kernels.c");
    println!("cargo:rerun-if-env-changed=NAM_DISABLE_PORTABLE_VECTOR_TANH");
    println!("cargo:rerun-if-env-changed=NAM_BUILD_GIT_COMMIT");
    emit_build_metadata();
    if std::env::var("CARGO_FEATURE_FAST_KERNELS").is_ok() {
        if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
            && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu")
            && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        {
            println!("cargo:rustc-link-lib=dl");
        }
        let mut build = cc::Build::new();
        build.file("csrc/fast_kernels.c").opt_level(3);
        if std::env::var("NAM_DISABLE_PORTABLE_VECTOR_TANH").is_ok() {
            build.define("NAM_DISABLE_PORTABLE_VECTOR_TANH", None);
        }
        build.compile("fast_kernels");
    }
}

fn emit_build_metadata() {
    let commit = std::env::var("NAM_BUILD_GIT_COMMIT")
        .ok()
        .filter(|value| valid_commit(value))
        .or_else(git_commit)
        .unwrap_or_else(|| "unknown".to_string());
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let features = ["float_io", "faer", "fast-kernels", "benchmark-internals"]
        .into_iter()
        .filter(|feature| {
            let variable = format!("CARGO_FEATURE_{}", feature.replace('-', "_").to_uppercase());
            std::env::var_os(variable).is_some()
        })
        .collect::<Vec<_>>()
        .join(",");
    let features = if features.is_empty() {
        "portable".to_string()
    } else {
        features
    };
    let short_commit = commit.get(..12).unwrap_or(&commit);
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_string());
    let summary = format!("{version} ({short_commit}; {target}; {profile}; {features})");

    println!("cargo:rustc-env=NAM_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=NAM_BUILD_TARGET={target}");
    println!("cargo:rustc-env=NAM_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=NAM_BUILD_FEATURES={features}");
    println!("cargo:rustc-env=NAM_BUILD_SUMMARY={summary}");
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?.trim().to_string();
    valid_commit(&commit).then_some(commit)
}

fn valid_commit(value: &str) -> bool {
    (7..=64).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
