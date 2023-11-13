use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::max_pool2d_i8::OpMaxPool2DInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_errors::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use core::f32::consts::PI;

const BUFFER: &[u8; 8856] =
    include_bytes!("../resources/models/simple_depthwise_conv_avg_pool_relu6_int8.tflite");

const ARENA_SIZE: usize = 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(
    interpreter: &mut BLiteInterpreter<'_, i8>,
    input_h: usize,
    input_w: usize,
    input: i8,
) {
    for h in 0..input_h {
        for w in 0..input_w {
            interpreter.input.data[h * input_w + w] = input;
        }
    }
}

fn predict() -> Result<()> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<6, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;
    op_resolver.add_op(OpReshapeInt8::reshape_int8())?;
    op_resolver.add_op(OpConv2DInt8::conv2d_int8())?;
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8())?;
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8())?;
    op_resolver.add_op(OpMaxPool2DInt8::max_pool2d_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;
    let (input_scale, input_zero_point) = interpreter.get_input_quantization_params().unwrap();
    let (output_scale, output_zero_point) = interpreter.get_output_quantization_params().unwrap();

    let mut golden_inputs_f32_inputs = vec![];
    for i in -128..127 {
        golden_inputs_f32_inputs.push((i, 0.1));
    }
    let delta = 0.08;
    for (g_input, _g_f32_input) in golden_inputs_f32_inputs {
        set_input(&mut interpreter, 6, 6, g_input);
        interpreter.invoke()?;

        let output = interpreter.output.data[0];
        let y_pred = (output as i32 - output_zero_point) as f32 * output_scale;
        let g_truth_input = (g_input as i32 - input_zero_point) as f32 * input_scale * PI;
        let g_truth_output = g_truth_input.sin();
        println!("input: {g_input:.8}, output: {output}, y_pred: {y_pred:.8}, ground truth input: {g_truth_input:.8} ground truth: {g_truth_output:.8}");
        if (y_pred - g_truth_output).abs() > delta {
            println!("Error!: abs :{}", (y_pred - g_truth_output).abs());
            // return Err(BLiteError::FatalError);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    predict()?;
    println!("Inference Success!!");
    Ok(())
}
