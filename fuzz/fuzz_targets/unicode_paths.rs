#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let mut input = Unstructured::new(data);
    let Ok((destination, paths, template)) =
        <(String, Vec<String>, Option<String>)>::arbitrary(&mut input)
    else {
        return;
    };
    let artifacts =
        nam_trainer::TrainingRunArtifacts::new(destination, &paths).with_naming(None, template);
    let _ = artifacts.predicted_model_paths();
    let _ = artifacts.duplicate_artifact_paths();
});
