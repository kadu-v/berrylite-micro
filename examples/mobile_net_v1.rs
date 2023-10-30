use berrylite::kernel::micro_operator::i8::avg_pool2d_i8::OpAvgPool2DInt8;
use berrylite::kernel::micro_operator::i8::conv2d_i8::OpConv2DInt8;
use berrylite::kernel::micro_operator::i8::depthwise_conv2d_i8::OpDepthWiseConv2DInt8;
use berrylite::kernel::micro_operator::i8::fully_connected_i8::OpFullyConnectedInt8;
use berrylite::kernel::micro_operator::i8::reshape_i8::OpReshapeInt8;
use berrylite::kernel::micro_operator::i8::softmax_i8::OpSoftMaxInt8;
use berrylite::micro_allocator::{ArenaAllocator, BumpArenaAllocator};
use berrylite::micro_erros::Result;
use berrylite::micro_interpreter::BLiteInterpreter;
use berrylite::micro_op_resolver::BLiteOpResolver;
use berrylite::tflite_schema_generated::tflite;

const BUFFER: &[u8; 1364512] =
    include_bytes!("../models/mobilenet_v1_0.50_192_quantized_1_default_1.tflite");

const ARENA_SIZE: usize = 1024 * 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn set_input(
    interpreter: &mut BLiteInterpreter<'_, i8>,
    input_h: usize,
    input_w: usize,
    _input_zero_point: i32,
) {
    for h in 0..input_h {
        for w in 0..input_w {
            interpreter.input.data[h * input_w + w] = 0 as i8;
        }
    }
}

fn predict() -> Result<usize> {
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
    println!("{:?}", allocator.description());

    set_input(&mut interpreter, 96, 96, input_zero_point);
    println!("inference start");
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

fn main() {
    // // ファイル展開とエラーチェック(.unwrap())

    // // グレースケール用のデータを作成
    // let mut gray_img = image::GrayImage::new(96, 96);

    // for y in 0..96 {
    //     for x in 0..96 {
    //         // let val = [(g_person_data[y * 96 + x] as i32 - 128) as u8];
    //         // ピクセルデータをあらかじめ作っておいたグレースケールデータに書き込む
    //         gray_img.put_pixel(x as u32, y as u32, image::Luma(val));
    //     }
    // }

    // 画像をファイルとして保存する。エラーチェックも忘れずに
    // gray_img.save("./gray.png").unwrap();

    // let images = [&g_person_data, &g_no_person_data];
    for _i in 0..1 {
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
}
