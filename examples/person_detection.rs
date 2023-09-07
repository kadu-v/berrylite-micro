use berrylite::kernel::micro_operator::f32::{
    conv2d::OpConv2D, depthwise_conv2d::OpDepthWiseConv2D, fully_connected::OpFullyConnected,
    max_pool2d::OpMaxPool2D, reshape::OpReshape, softmax::OpSoftMax,
};
use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::max_pool2d_i8::OpMaxPool2DInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;

const BUFFER: &[u8; 300568] = include_bytes!("../models/person_detect.tflite");

const ARENA_SIZE: usize = 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(interpreter: &mut BLiteInterpreter<'_, i8>, input_h: usize, input_w: usize) {
    for h in 0..input_h {
        for w in 0..input_w {
            interpreter.input.data[h * input_w + w] = 0;
        }
    }
}

fn predict() -> Result<usize> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<6, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;
    op_resolver.add_op(OpReshapeInt8::reshape_int8())?;
    op_resolver.add_op(OpConv2DInt8::conv2d_int8())?;
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8())?;
    // op_resolver.add_op(OpSoftMax::softmax())?;
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    println!("{:?}", allocator.description());

    set_input(&mut interpreter, 28, 28);
    interpreter.invoke()?;

    let output = interpreter.output;
    let mut num_prob = 0.;
    let mut num = 0;
    // for (i, &prob) in output.data.iter().enumerate() {
    //     if prob > num_prob {
    //         num_prob = prob;
    //         num = i;
    //     }
    // }
    dbg!(&output.data);
    Ok(num)
}

fn main() {
    let y_pred = match predict() {
        Ok(y_pred) => y_pred,
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
    };
    println!("number: {}", y_pred);
    println!("Inference Success!!");
}
