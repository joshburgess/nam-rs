use crate::error::NamError;

/// Verify that the config version is supported.
/// Accepts versions from 0.5.0 through 0.7.x.
/// Versions above 0.7.x with same major produce a warning but still succeed.
/// Versions below 0.5.0 or with major != 0 fail.
pub fn verify_config_version(version_str: &str) -> Result<(), NamError> {
    let parts: Vec<&str> = version_str.split('.').collect();
    if parts.len() < 2 {
        return Err(NamError::UnsupportedVersion(version_str.to_string()));
    }
    let major: u32 = parts[0]
        .parse()
        .map_err(|_| NamError::UnsupportedVersion(version_str.to_string()))?;
    let minor: u32 = parts[1]
        .parse()
        .map_err(|_| NamError::UnsupportedVersion(version_str.to_string()))?;

    if major != 0 {
        return Err(NamError::UnsupportedVersion(version_str.to_string()));
    }
    if minor < 5 {
        return Err(NamError::UnsupportedVersion(version_str.to_string()));
    }
    // 0.5.x through 0.7.x are fully supported
    if minor > 7 {
        eprintln!(
            "Warning: Model config version {} is newer than latest fully-supported version 0.7.0. \
             Continuing with partial support.",
            version_str
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_versions() {
        assert!(verify_config_version("0.5.0").is_ok());
        assert!(verify_config_version("0.5.4").is_ok());
        assert!(verify_config_version("0.5.100").is_ok());
        assert!(verify_config_version("0.6.0").is_ok());
        assert!(verify_config_version("0.6.1").is_ok());
        assert!(verify_config_version("0.7.0").is_ok());
        assert!(verify_config_version("0.7.5").is_ok());
    }

    #[test]
    fn test_invalid_versions() {
        assert!(verify_config_version("1.0.0").is_err());
        assert!(verify_config_version("0.4.0").is_err());
        assert!(verify_config_version("0.4.9").is_err());
        assert!(verify_config_version("bad").is_err());
    }

    #[test]
    fn test_future_versions_warn_but_succeed() {
        // Versions > 0.7.x should succeed (with warning)
        assert!(verify_config_version("0.8.0").is_ok());
        assert!(verify_config_version("0.9.0").is_ok());
    }

    #[test]
    fn test_empty_version_string() {
        assert!(verify_config_version("").is_err());
    }

    #[test]
    fn test_two_part_version() {
        assert!(verify_config_version("0.5").is_ok());
        assert!(verify_config_version("0.6").is_ok());
        assert!(verify_config_version("0.7").is_ok());
    }

    #[test]
    fn test_four_part_version() {
        assert!(verify_config_version("0.5.4.1").is_ok());
    }

    #[test]
    fn test_non_numeric_parts() {
        assert!(verify_config_version("a.b.c").is_err());
        assert!(verify_config_version("0.b.0").is_err());
    }

    #[test]
    fn test_single_number() {
        assert!(verify_config_version("5").is_err());
    }
}
