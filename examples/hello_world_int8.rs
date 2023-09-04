use berrylite::kernel::micro_operator::u8::fully_connected_u8::OpFullyConnectedInt8;
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use core::f32::consts::PI;

const BUFFER: &[u8; 2704] = include_bytes!("../models/hello_world_int8.tflite");

const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(interpreter: &mut BLiteInterpreter<'_, i8>, input: i8) {
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

    let delta = 0.05;
    let golden_inputs_f32_inputs = [(-96, 0.77f32), (-63, 1.57), (-34, 2.3), (0, 3.14)];
    for (g_input, g_f32_input) in golden_inputs_f32_inputs {
        let input = g_input;
        println!(
            "[Input]: {} {} {} {}",
            g_input,
            input_scale,
            input_zero_point,
            input_scale * (g_input - input_zero_point) as f32
        );
        set_input(&mut interpreter, input as i8);
        interpreter.invoke()?;
        let output = interpreter.output.data[0];
        let y_pred = (output as i32 - output_zero_point) as f32 * output_scale;
        let g_truth_input = input_scale * ((input as i8) as i32 - input_zero_point) as f32;
        let g_truth_output = g_truth_input.sin();
        println!("input: {input:.8}, y_pred: {y_pred:.8}, ground truth input: {g_truth_input:.8} ground truth: {g_truth_output:.8}");
        if (y_pred - g_truth_output).abs() > delta {
            println!("Error!: abs :{}", (y_pred - g_truth_output).abs());
        }
    }

    Ok(())
}

fn main() {
    predict();
    println!("Inference Success!!");
}
