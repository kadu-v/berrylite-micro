use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::kernel::micro_operator::i8::softmax_i8::OpSoftMaxInt8;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};

criterion_group!(
    benches,
    benchmark_mobilenet_v1_0_50_128_quantized_1_default_1,
    benchmark_mobilenet_v1_0_50_160_quantized_1_default_1,
    benchmark_mobilenet_v1_0_50_192_quantized_1_default_1,
    benchmark_mobilenet_v1_0_50_224_quantized_1_default_1,
);

criterion_main!(benches);

const ARENA_SIZE: usize = 10 * 1024 * 1024;
static mut BASE_ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

const MOBILENET_V1_0_50_128_QUANTIZED: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_128_quantized_1_default_1.tflite");

#[allow(unused_must_use)]
pub fn benchmark_mobilenet_v1_0_50_128_quantized_1_default_1(c: &mut Criterion) {
    let base_model = tflite::root_as_model(MOBILENET_V1_0_50_128_QUANTIZED).unwrap();
    let mut base_allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut base_op_resolver = BLiteOpResolver::<7, i8, _>::new();
    base_op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    base_op_resolver.add_op(OpReshapeInt8::reshape_int8());
    base_op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    base_op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    base_op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    base_op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let base_interpreter =
        BLiteInterpreter::new(&mut base_allocator, &base_op_resolver, &base_model).unwrap();

    let model = tflite::root_as_model(MOBILENET_V1_0_50_128_QUANTIZED).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut BASE_ARENA) };
    let mut op_resolver = BLiteOpResolver::<7, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    op_resolver.add_op(OpReshapeInt8::reshape_int8());
    op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model).unwrap();

    let mut group = c.benchmark_group("mobilenet_v1_0_50_128_quantized");
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function("base", |b| {
        b.iter(|| {
            base_interpreter.invoke();
        })
    });
    group.bench_function("pruning", |b| {
        b.iter(|| {
            interpreter.invoke();
        })
    });
    group.finish();
}

const MOBILENET_V1_0_50_160_QUANTIZED: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_160_quantized_1_default_1.tflite");

#[allow(unused_must_use)]
pub fn benchmark_mobilenet_v1_0_50_160_quantized_1_default_1(c: &mut Criterion) {
    let base_model = tflite::root_as_model(MOBILENET_V1_0_50_160_QUANTIZED).unwrap();
    let mut base_allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut base_op_resolver = BLiteOpResolver::<7, i8, _>::new();
    base_op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    base_op_resolver.add_op(OpReshapeInt8::reshape_int8());
    base_op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    base_op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    base_op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    base_op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let base_interpreter =
        BLiteInterpreter::new(&mut base_allocator, &base_op_resolver, &base_model).unwrap();

    let model = tflite::root_as_model(MOBILENET_V1_0_50_160_QUANTIZED).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut BASE_ARENA) };
    let mut op_resolver = BLiteOpResolver::<7, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    op_resolver.add_op(OpReshapeInt8::reshape_int8());
    op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model).unwrap();

    let mut group = c.benchmark_group("mobilenet_v1_0_50_160_quantized");
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function("base", |b| {
        b.iter(|| {
            base_interpreter.invoke();
        })
    });
    group.bench_function("pruning", |b| {
        b.iter(|| {
            interpreter.invoke();
        })
    });
    group.finish();
}

const MOBILENET_V1_0_50_192_QUANTIZED: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_192_quantized_1_default_1.tflite");

#[allow(unused_must_use)]
pub fn benchmark_mobilenet_v1_0_50_192_quantized_1_default_1(c: &mut Criterion) {
    let base_model = tflite::root_as_model(MOBILENET_V1_0_50_192_QUANTIZED).unwrap();
    let mut base_allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut base_op_resolver = BLiteOpResolver::<7, i8, _>::new();
    base_op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    base_op_resolver.add_op(OpReshapeInt8::reshape_int8());
    base_op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    base_op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    base_op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    base_op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let base_interpreter =
        BLiteInterpreter::new(&mut base_allocator, &base_op_resolver, &base_model).unwrap();

    let model = tflite::root_as_model(MOBILENET_V1_0_50_192_QUANTIZED).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut BASE_ARENA) };
    let mut op_resolver = BLiteOpResolver::<7, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    op_resolver.add_op(OpReshapeInt8::reshape_int8());
    op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model).unwrap();

    let mut group = c.benchmark_group("mobilenet_v1_0_50_192_quantized");
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function("base", |b| {
        b.iter(|| {
            base_interpreter.invoke();
        })
    });
    group.bench_function("pruning", |b| {
        b.iter(|| {
            interpreter.invoke();
        })
    });
    group.finish();
}

const MOBILENET_V1_0_50_224_QUANTIZED: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_224_quantized_1_default_1.tflite");

#[allow(unused_must_use)]
pub fn benchmark_mobilenet_v1_0_50_224_quantized_1_default_1(c: &mut Criterion) {
    let base_model = tflite::root_as_model(MOBILENET_V1_0_50_224_QUANTIZED).unwrap();
    let mut base_allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut base_op_resolver = BLiteOpResolver::<7, i8, _>::new();
    base_op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    base_op_resolver.add_op(OpReshapeInt8::reshape_int8());
    base_op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    base_op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    base_op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    base_op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let base_interpreter =
        BLiteInterpreter::new(&mut base_allocator, &base_op_resolver, &base_model).unwrap();

    let model = tflite::root_as_model(MOBILENET_V1_0_50_224_QUANTIZED).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut BASE_ARENA) };
    let mut op_resolver = BLiteOpResolver::<7, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    op_resolver.add_op(OpReshapeInt8::reshape_int8());
    op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());
    let interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model).unwrap();

    let mut group = c.benchmark_group("mobilenet_v1_0_50_224_quantized");
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function("base", |b| {
        b.iter(|| {
            base_interpreter.invoke();
        })
    });
    group.bench_function("pruning", |b| {
        b.iter(|| {
            interpreter.invoke();
        })
    });
    group.finish();
}
