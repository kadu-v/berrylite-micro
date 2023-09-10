use criterion::{criterion_group, criterion_main};
use mobilenet_v1::bm_mobilenet_v1_0_50_192_quantized_1_default_1;

pub mod mobilenet_v1;

criterion_group!(benches, bm_mobilenet_v1_0_50_192_quantized_1_default_1);
criterion_main!(benches);
