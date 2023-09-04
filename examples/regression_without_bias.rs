use berrylite::kernel::micro_operator::u8::fully_connected_u8::OpFullyConnectedInt8;
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;

// 2 * x
//----good----
// const BUFFER: &[u8; 1112] = include_bytes!("../models/regression_without_bias_int8_1layer.tflite");
// const BUFFER: &[u8; 1112] =
// include_bytes!("../models/regression_without_bias_int8_1layer_2units.tflite");
// const BUFFER: &[u8; 1424] =
//     include_bytes!("../models/regression_without_bias_int8_2layer_1units.tflite");
// const BUFFER: &[u8; 1440] = include_bytes!("../models/regression_without_bias_int8_2-1.tflite");

//----not good----
const BUFFER: &[u8; 1440] = include_bytes!("../models/regression_without_bias_int8_1-2.tflite");
// const BUFFER: &[u8; 1424] = include_bytes!("../models/regression_without_bias_int8_2layer.tflite");
// const BUFFER: &[u8; 1488] = include_bytes!("../models/regression_without_bias_int8.tflite");

const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(interpreter: &mut BLiteInterpreter<'_, u8>, input: u8) {
    interpreter.input.data[0] = input;
}

fn predict() -> Result<()> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<1, u8>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    let (input_scale, input_zero_point) = interpreter.get_input_quantization_params().unwrap();
    let (output_scale, output_zero_point) = interpreter.get_output_quantization_params().unwrap();

    let delta = 0.0001;
    let mut golden_inputs_f32_inputs = vec![];
    for i in -127..128 {
        golden_inputs_f32_inputs.push(i);
    }
    for g_input in golden_inputs_f32_inputs {
        let input = g_input;
        let input = input as u8;

        set_input(&mut interpreter, input as u8);
        interpreter.invoke()?;
        let output = interpreter.output.data[0];
        let y_pred = (output as i32 - output_zero_point) as f32 * output_scale;
        let g_truth_input = input_scale * ((input as u8) as i32 - input_zero_point) as f32;
        let g_truth_output = 2. * g_truth_input as f32;
        println!("zero_point: {input_zero_point}, input: {input:.8}, y_pred: {y_pred:.8}, g_input: {g_truth_input}, ground truth: {g_truth_output:.8}");
        // if (y_pred - g_truth_output).abs() > delta {
        //     println!("Error!: abs :{}", (y_pred - g_truth_output).abs());
        // }
    }
    println!("[Input]: {} {}", input_scale, input_zero_point);

    Ok(())
}

fn main() {
    predict();
    println!("Inference Success!!");
}
