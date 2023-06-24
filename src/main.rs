use std::mem::size_of;

use berrylite::tflite_schema_generated::tflite;
use flatbuffers;
const BUFFER: &[u8; 3164] = include_bytes!("../models/hello_world_float.tflite");
// const BUFFER: &[u8; 300568] = include_bytes!("../models/person_detect.tflite");

fn main() {
    let model = tflite::root_as_model(BUFFER).unwrap();
    println!("model version: {}", model.version());
    let subgraphs = model.subgraphs().unwrap();
    let subgraph = subgraphs.get(0);
    println!("subgraphs size: {}", subgraphs.len());
    println!("subgraph :{:?}", subgraph.operators().unwrap().len());

    let tensors = subgraph.tensors().unwrap();
    // println!("{:?}", tensors);
    let buffers = model.buffers().unwrap();
    // println!("{:?}", buffers);

    let tensor = tensors.get(0);
    println!("{:?}", tensor);

    let buffer = buffers.get(2);
    println!("{:?}", buffer);

    let data = buffer.data().unwrap();
    println!("data: {:?}", data);

    let row_data = data.bytes();
    println!("{:?}, len: {}", row_data, row_data.len());

    unsafe {
        let v = core::slice::from_raw_parts(
            row_data.as_ptr() as *const f32,
            row_data.len() / size_of::<f32>(),
        );
        println!("{:?}", v);
        println!("{}", v.len());
    }
    // for (i, tensor) in tensors.iter().enumerate() {
    //     let buffer = buffers.get(tensor.buffer() as usize);
    //     println!("{}", i);
    //     println!("tensor: {:?}", tensor);
    //     println!(
    //         "{}: {:?}\n",
    //         buffer.data().unwrap_or_default().len(),
    //         buffer
    //     );
    // }

    // let op = subgraph.operators().unwrap().get(0);
    // println!("{:?}", op);
}
