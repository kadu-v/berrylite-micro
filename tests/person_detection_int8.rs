use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::kernel::micro_operator::i8::softmax_i8::OpSoftMaxInt8;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_errors::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use image::{ImageBuffer, Luma};

const BUFFER: &[u8; 300568] = include_bytes!("../resources/models/person_detect.tflite");

const ARENA_SIZE: usize = 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(
    interpreter: &mut BLiteInterpreter<'_, i8>,
    input_h: usize,
    input_w: usize,
    _input_zero_point: i32,
    image: &[u8],
) {
    for h in 0..input_h {
        for w in 0..input_w {
            let v = image[h * input_w + w];
            interpreter.input.data[h * input_w + w] = v as i8;
        }
    }
}

fn predict(image: &Vec<u8>) -> Result<usize> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<7, _, _>::new();
    op_resolver.add_op(OpFullyConnectedInt8::fully_connected_int8())?;
    op_resolver.add_op(OpReshapeInt8::reshape_int8())?;
    op_resolver.add_op(OpConv2DInt8::conv2d_int8())?;
    op_resolver.add_op(OpAvgPool2DInt8::avg_pool2d_int8())?;
    op_resolver.add_op(OpSoftMaxInt8::softmax_int8())?;
    op_resolver.add_op(OpDepthWiseConv2DInt8::depthwise_conv2d_int8())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;
    let (_input_scale, input_zero_point) = interpreter.get_input_quantization_params().unwrap();
    let (output_scale, output_zero_point) = interpreter.get_output_quantization_params().unwrap();

    set_input(&mut interpreter, 96, 96, input_zero_point, image);
    interpreter.invoke()?;

    let output = interpreter.output;
    dbg!(output);
    let mut num_prob = 0.;
    let mut num = 0;
    for (i, &y_pred) in output.data.iter().enumerate() {
        let prob = output_scale * (y_pred as i32 - output_zero_point) as f32;
        dbg!(prob);
        if prob > num_prob {
            num_prob = prob;
            num = i;
        }
    }
    Ok(num)
}

fn make_vec_from_image(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<u8> {
    let mut v = Vec::new();
    for h in 0..96 {
        for w in 0..96 {
            let p = img.get_pixel(w, h).0[0];
            v.push(p);
        }
    }
    v
}

#[test]
fn test_person_detection_int8() {
    let inputs = [
        ("person0", 0),
        ("person1", 0),
        ("mario0", 0),
        ("dog0", 1),
        ("cat0", 1),
    ];

    for (img_name, expected) in inputs {
        let img_path = format!("./resources/dataset/person_detection/{}.jpg", img_name);
        let img = image::open(&img_path).unwrap();
        dbg!(img.height(), img.width());
        let img = img.into_luma8();
        let input = make_vec_from_image(&img);
        let y_pred = match predict(&input) {
            Ok(y_pred) => y_pred,
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        };
        assert_eq!(
            y_pred, expected,
            "the path of an input image is \"{}\"",
            img_path
        );
    }
}
