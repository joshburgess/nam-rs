#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    if let Ok(name) = std::str::from_utf8(data) {
        let sanitized = nam_trainer::sanitize_model_basename(name);
        assert!(!sanitized.is_empty());
    }
});
