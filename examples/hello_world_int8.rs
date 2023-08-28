use berrylite::kernel::micro_operator::u8::fully_connected_u8::OpFullyConnectedInt8;
use berrylite::micro_allocator::{BumpArenaAllocator, ArenaAllocator};
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use core::f32::consts::PI;

const BUFFER: &[u8; 2704] =
    include_bytes!("../models/hello_world_int8.tflite");

const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(
    interpreter: &mut BLiteInterpreter<'_, u8>,
    input: u8,
) {
    interpreter.input.data[0] = input;
}

fn predict(input: u8) -> Result<u8> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator =
        unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<1, u8>::new();
    op_resolver.add_op(
        OpFullyConnectedInt8::fully_connected_int8(),
    )?;

    let mut interpreter = BLiteInterpreter::new(
        &mut allocator,
        &op_resolver,
        &model,
    )?;

    println!("{:?}", allocator.description());
    set_input(&mut interpreter, input);
    interpreter.invoke()?;

    let output = interpreter.output;

    Ok(output.data[0])
}

fn main() {
    let delta = 0.05;
    let inputs = [1];
    for input in inputs {
        // let input = input * PI
        let y_pred = match predict(input) {
            Ok(y_pred) => y_pred,
            Err(e) => {
                println!("Error: {:?}", e);
                return;
            }
        };
        // let ground_truth = input.sin();
        // println!("input: {input:.8}, y_pred: {y_pred:.8}, ground truth: {ground_truth:.8}");
        // if (y_pred - ground_truth).abs() > delta {
        //     println!(
        //         "Error!: abs :{}",
        //         (y_pred - ground_truth).abs()
        //     );
        //     return;
        // }
    }
    println!("Inference Success!!");
}
