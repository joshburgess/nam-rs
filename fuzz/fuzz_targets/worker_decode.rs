#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use serde_json::json;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    if let Ok(json) = std::str::from_utf8(data) {
        let _ = nam_trainer::decode_worker_event_json(json);
    }

    let mut input = Unstructured::new(data);
    let Ok((kind, sequence, file_index, message)) =
        <(u8, u64, Option<u8>, String)>::arbitrary(&mut input)
    else {
        return;
    };
    let file_index = file_index.map(usize::from);
    let event = match kind % 4 {
        0 => json!({
            "type": "training_start",
            "protocol_version": nam_trainer::PROTOCOL_VERSION,
            "run_id": "structured-run",
            "file_index": file_index,
            "sequence": sequence,
            "file": message,
            "total_epochs": 1
        }),
        1 => json!({
            "type": "epoch_end",
            "protocol_version": nam_trainer::PROTOCOL_VERSION,
            "run_id": "structured-run",
            "file_index": file_index,
            "sequence": sequence,
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "esr": 0.3
        }),
        2 => json!({
            "type": "training_complete",
            "protocol_version": nam_trainer::PROTOCOL_VERSION,
            "run_id": "structured-run",
            "file_index": file_index,
            "sequence": sequence,
            "file": message,
            "validation_esr": 0.25,
            "model_path": "structured.nam"
        }),
        _ => json!({
            "type": "log",
            "protocol_version": nam_trainer::PROTOCOL_VERSION,
            "run_id": "structured-run",
            "file_index": file_index,
            "sequence": sequence,
            "message": message
        }),
    };
    assert!(nam_trainer::decode_worker_event_json(&event.to_string()).is_ok());
});
