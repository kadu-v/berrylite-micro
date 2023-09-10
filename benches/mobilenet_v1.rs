use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::kernel::micro_operator::i8::softmax_i8::OpSoftMaxInt8;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;

use criterion::Criterion;

const BUFFER: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_192_quantized_1_default_1.tflite");

const ARENA_SIZE: usize = 1024 * 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

#[allow(unused_must_use)]
pub fn bm_mobilenet_v1_0_50_192_quantized_1_default_1(c: &mut Criterion) {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<7, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8());
    op_resolver.add_op(OpReshapeInt8::reshape_int8());
    op_resolver.add_op(OpConv2DInt8::conv2d_int8());
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8());
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8());
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8());

    let interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model).unwrap();
    c.bench_function("mobilent_v1_0_50_193_quantized_1_defualt", |b| {
        b.iter(|| interpreter.invoke())
    });
}
