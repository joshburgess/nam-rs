pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_COMMIT: &str = env!("NAM_BUILD_GIT_COMMIT");
pub const TARGET: &str = env!("NAM_BUILD_TARGET");
pub const PROFILE: &str = env!("NAM_BUILD_PROFILE");
pub const FEATURES: &str = env!("NAM_BUILD_FEATURES");
pub const SUMMARY: &str = env!("NAM_BUILD_SUMMARY");

pub fn json() -> String {
    serde_json::json!({
        "version": VERSION,
        "git_commit": GIT_COMMIT,
        "target": TARGET,
        "profile": PROFILE,
        "features": FEATURES.split(',').collect::<Vec<_>>(),
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_build_metadata_is_complete() {
        assert!(!VERSION.is_empty());
        assert!(!GIT_COMMIT.is_empty());
        assert!(!TARGET.is_empty());
        assert!(!PROFILE.is_empty());
        assert!(!FEATURES.is_empty());
        assert!(serde_json::from_str::<serde_json::Value>(&json()).is_ok());
    }
}
