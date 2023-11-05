use berrylite::kernel::micro_operator::f32::fully_connected::OpFullyConnected;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use core::f32::consts::PI;
use float_eq::assert_float_eq;

const BUFFER: &[u8; 3164] = include_bytes!("../models/hello_world_float.tflite");

const ARENA_SIZE: usize = 100 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(interpreter: &mut BLiteInterpreter<'_, f32>, input: f32) {
    interpreter.input.data[0] = input;
}

fn predict(input: f32) -> Result<f32> {
    let model = tflite::root_as_model(BUFFER).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut op_resolver = BLiteOpResolver::<1, f32, _>::new();
    op_resolver.add_op(OpFullyConnected::fully_connected())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    set_input(&mut interpreter, input);
    interpreter.invoke()?;
    let output = interpreter.output;

    Ok(output.data[0])
}

#[test]
fn test_hello_world() {
    let delta = 0.03;
    let inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        .into_iter()
        .map(|x| x * PI);
    let expected_outputs = inputs
        .clone()
        .into_iter()
        .map(|x| x.sin())
        .collect::<Vec<f32>>();

    for (i, input) in inputs.enumerate() {
        let y_pred = predict(input).unwrap();
        let expected = expected_outputs[i];
        assert_float_eq!(y_pred, expected, abs <= delta);
    }
}
