use nam_plugin::benchmark::CallbackCase;
use std::hint::black_box;

fn main() -> Result<(), nam_core::NamError> {
    let mut case = CallbackCase::new_a1(64)?;
    for _ in 0..128 {
        case.process();
    }
    black_box(case);
    Ok(())
}
