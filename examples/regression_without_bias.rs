use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_array::ArrayElem;
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
// const BUFFER: &[u8; 1480] = include_bytes!("../models/regression_without_bias_10_1.tflite");

const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input<T: ArrayElem<T>>(interpreter: &mut BLiteInterpreter<'_, T>, input: T) {
    interpreter.input.data[0] = input;
}

fn predict() -> Result<()> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<1, i8>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    let (input_scale, input_zero_point) = interpreter.get_input_quantization_params().unwrap();
    let (output_scale, output_zero_point) = interpreter.get_output_quantization_params().unwrap();

    let delta = 0.007;
    let mut g_inputs = vec![];
    for i in -128..=127 {
        let f32_val = input_scale * (i - input_zero_point) as f32;
        g_inputs.push((i, f32_val));
    }

    for (input, g_input) in g_inputs {
        let input = input as i8;
        set_input(&mut interpreter, input as i8);
        interpreter.invoke()?;
        let output = interpreter.output.data[0];
        let y_pred = (output as i32 - output_zero_point) as f32 * output_scale;
        let g_truth_output = 2. * g_input;
        println!("zero_point: {input_zero_point}, input: {input:.8}, y_pred: {y_pred:.8}, g_input: {g_input}, ground truth: {g_truth_output:.8}");
        if (y_pred - g_truth_output).abs() > delta {
            println!("[Error!]: abs :{}", (y_pred - g_truth_output).abs());
        }
    }
    println!("[Input]: {} {}", input_scale, input_zero_point);

    Ok(())
}

fn main() {
    match predict() {
        Ok(_) => {
            println!("Inference Success!!");
        }
        Err(e) => {
            println!("[Error!]: {:?}", e);
        }
    }
}
