#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::path::Path;

#[derive(Arbitrary, Debug)]
enum Operation {
    RestorePacked {
        sample_rate: u8,
        max_buffer_size: u16,
    },
    RestoreA2 {
        sample_rate: u8,
        max_buffer_size: u16,
    },
    Clear {
        sample_rate: u8,
        max_buffer_size: u16,
    },
    RestoreSerializedPath {
        serialized: String,
        sample_rate: u8,
        max_buffer_size: u16,
    },
    SetModelSize(u32),
    Process(u16),
    Reset,
    Deactivate,
}

fn sample_rate(selector: u8) -> f32 {
    match selector % 6 {
        0 => 0.0,
        1 => f32::NAN,
        2 => 44_100.0,
        3 => 48_000.0,
        4 => 96_000.0,
        _ => 768_000.0,
    }
}

fn buffer_size(raw: u16) -> u32 {
    match raw % 8 {
        0 => 0,
        1 => 1,
        2 => 16,
        3 => 64,
        4 => 257,
        5 => 4096,
        6 => 8192,
        _ => 1_048_577,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let mut unstructured = Unstructured::new(data);
    let Ok(operations) = Vec::<Operation>::arbitrary(&mut unstructured) else {
        return;
    };
    let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../nam-core/test_fixtures/models");
    let packed = fixture_root.join("upstream_packed_a2_export.nam");
    let a2 = fixture_root.join("wavenet_a2_max.nam");
    let mut lifecycle = nam_plugin::benchmark::LifecycleCase::default();
    assert!(lifecycle.restore_model(&packed, 44_100.0, 257));
    lifecycle.set_model_size(0.0);
    assert!(lifecycle.process(16));
    lifecycle.set_model_size(0.5);
    assert!(lifecycle.process(257));
    lifecycle.set_model_size(1.0);
    lifecycle.reset();
    assert!(lifecycle.restore_model(&a2, 96_000.0, 64));
    assert!(lifecycle.process(64));
    assert!(lifecycle.clear_model(48_000.0, 128));

    for operation in operations.into_iter().take(32) {
        match operation {
            Operation::RestorePacked {
                sample_rate: rate,
                max_buffer_size,
            } => {
                let _ = lifecycle.restore_model(
                    &packed,
                    sample_rate(rate),
                    buffer_size(max_buffer_size),
                );
            }
            Operation::RestoreA2 {
                sample_rate: rate,
                max_buffer_size,
            } => {
                let _ = lifecycle.restore_model(
                    &a2,
                    sample_rate(rate),
                    buffer_size(max_buffer_size),
                );
            }
            Operation::Clear {
                sample_rate: rate,
                max_buffer_size,
            } => {
                let _ = lifecycle.clear_model(
                    sample_rate(rate),
                    buffer_size(max_buffer_size),
                );
            }
            Operation::RestoreSerializedPath {
                serialized,
                sample_rate: rate,
                max_buffer_size,
            } => {
                let _ = lifecycle.restore_serialized_model_path(
                    serialized,
                    sample_rate(rate),
                    buffer_size(max_buffer_size),
                );
            }
            Operation::SetModelSize(bits) => {
                lifecycle.set_model_size(f32::from_bits(bits));
            }
            Operation::Process(size) => {
                assert!(lifecycle.process(usize::from(size % 4097)));
            }
            Operation::Reset => lifecycle.reset(),
            Operation::Deactivate => lifecycle.deactivate(),
        }
    }
});
