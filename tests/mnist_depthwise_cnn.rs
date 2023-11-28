use berrylite::kernel::micro_operator::f32::{
    conv2d::OpConv2D, depthwise_conv2d::OpDepthWiseConv2D, fully_connected::OpFullyConnected,
    max_pool2d::OpMaxPool2D, reshape::OpReshape, softmax::OpSoftMax,
};
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_errors::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;
use image::{ImageBuffer, Luma};

const BUFFER: &[u8; 419572] = include_bytes!("../resources/models/mnist_depthwise_cnn.tflite");

const ARENA_SIZE: usize = 100 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(
    interpreter: &mut BLiteInterpreter<'_, f32>,
    input_h: usize,
    input_w: usize,
    image: &Vec<u8>,
) {
    for h in 0..input_h {
        for w in 0..input_w {
            interpreter.input.data[h * input_w + w] = image[h * input_w + w] as f32 / 255.;
        }
    }
}

fn predict(input: &Vec<u8>) -> Result<usize> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<6, f32, _>::new();
    op_resolver.add_op(OpFullyConnected::fully_connected())?;
    op_resolver.add_op(OpReshape::reshape())?;
    op_resolver.add_op(OpConv2D::conv2d())?;
    op_resolver.add_op(OpMaxPool2D::max_pool2d())?;
    op_resolver.add_op(OpSoftMax::softmax())?;
    op_resolver.add_op(OpDepthWiseConv2D::depthwise_conv2d())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

    println!("{:?}", allocator.description());

    set_input(&mut interpreter, 28, 28, input);
    interpreter.invoke()?;

    let output = interpreter.output;
    let mut num_prob = 0.;
    let mut num = 0;
    for (i, &prob) in output.data.iter().enumerate() {
        if prob > num_prob {
            num_prob = prob;
            num = i;
        }
    }
    dbg!(&output.data);
    Ok(num)
}

fn make_vec_from_image(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<u8> {
    let mut v = Vec::new();
    for h in 0..28 {
        for w in 0..28 {
            let p = img.get_pixel(w, h).0[0];
            v.push(p);
        }
    }
    v
}

#[test]
fn test_mnist_depthwise_cnn() {
    let inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    for expected in inputs {
        let img_path = format!("./resources/dataset/mnist/{}.jpg", expected);
        let img = image::open(&img_path).unwrap().into_luma8();
        let input = make_vec_from_image(&img);
        let y_pred = match predict(&input) {
            Ok(y_pred) => y_pred,
            Err(e) => {
                println!("Error: {:?}", e);
                return;
            }
        };
        assert_eq!(
            y_pred, expected,
            "the path of an input image is \"{}\"",
            img_path
        );
    }
}
