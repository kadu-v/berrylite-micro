use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_errors::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use float_eq::assert_float_eq;

const BUFFER: &[u8; 2704] = include_bytes!("../resources/models/hello_world_int8.tflite");

const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(interpreter: &mut BLiteInterpreter<'_, i8>, input: i8) {
    interpreter.input.data[0] = input;
}

fn predict(input: f32) -> Result<f32> {
    let model = tflite::root_as_model(BUFFER).unwrap();
    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };
    let mut op_resolver = BLiteOpResolver::<1, i8, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    let (input_scale, input_zero_point) = interpreter.get_input_quantization_params().unwrap();
    let (output_scale, output_zero_point) = interpreter.get_output_quantization_params().unwrap();

    let i8_input = (input / input_scale + input_zero_point as f32) as i8;
    set_input(&mut interpreter, i8_input);
    interpreter.invoke()?;

    let output = interpreter.output.data[0];
    let y_pred = (output as i32 - output_zero_point) as f32 * output_scale;

    Ok(y_pred)
}

#[test]
fn test_hello_world_int8() {
    let delta = 0.02;
    let inputs = [0.77f32, 1.57, 2.3, 3.14];
    let expected_outputs = inputs
        .clone()
        .into_iter()
        .map(|x| x.sin())
        .collect::<Vec<f32>>();

    for (i, input) in inputs.into_iter().enumerate() {
        let y_pred = predict(input).unwrap();
        let expected = expected_outputs[i];
        assert_float_eq!(y_pred, expected, abs <= delta);
    }
}
