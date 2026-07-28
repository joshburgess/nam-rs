#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use serde_json::json;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    if let Ok(json) = std::str::from_utf8(data) {
        let _ = nam_core::get_dsp::get_dsp_from_json(json);
    }

    let mut input = Unstructured::new(data);
    let Ok((raw_receptive_field, bias, raw_weights, raw_samples)) =
        <(u8, bool, Vec<i16>, Vec<i16>)>::arbitrary(&mut input)
    else {
        return;
    };
    let receptive_field = usize::from(raw_receptive_field % 32) + 1;
    let weight_count = receptive_field + usize::from(bias);
    let weights = raw_weights
        .into_iter()
        .chain(std::iter::repeat(0))
        .take(weight_count)
        .map(|weight| f64::from(weight) / f64::from(i16::MAX))
        .collect::<Vec<_>>();
    let model_json = json!({
        "version": "0.7.0",
        "architecture": "Linear",
        "config": {
            "receptive_field": receptive_field,
            "bias": bias
        },
        "weights": weights
    });
    let Ok(mut model) = nam_core::get_dsp::get_dsp_from_json(&model_json.to_string()) else {
        panic!("structured linear model must load");
    };
    let samples = raw_samples
        .into_iter()
        .take(4096)
        .map(|sample| nam_core::dsp::sample_from_f64(f64::from(sample) / f64::from(i16::MAX)))
        .collect::<Vec<_>>();
    let mut output = vec![nam_core::Sample::default(); samples.len()];
    model.reset(48_000.0, samples.len().max(1));
    model.process(&samples, &mut output);
    assert!(
        output
            .iter()
            .all(|sample| nam_core::dsp::sample_to_f64(*sample).is_finite())
    );
});
