use std::mem::size_of;

use berrylite::micro_allocator::ArenaAllocator;
use berrylite::micro_array::BLiteArray;
use berrylite::micro_graph::*;
use berrylite::tflite_schema_generated::tflite;
use flatbuffers;
const BUFFER: &[u8; 3164] = include_bytes!("../models/hello_world_float.tflite");
// const BUFFER: &[u8; 300568] = include_bytes!("../models/person_detect.tflite");

static mut ARENA: [u8; 3000] = [0; 3000];

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
    println!("data: {:?}", buffer);

    let row_data = data.bytes();
    println!("{:?}, len: {}", row_data, row_data.len());

    unsafe {
        let dims = &[10];
        let v = BLiteArray::<f32>::from_buffer(buffer, dims);
        println!("{:?}", v.as_ref().unwrap());
        println!("{}", v.as_ref().unwrap().len());
    }
    unsafe {
        let mut allocator = ArenaAllocator::new(&mut ARENA);
        let v = &mut *(allocator.alloc::<[f32; 10]>(1).unwrap() as *mut [f32; 10]);

        for i in 0..10 {
            v[i] = 0.1;
        }

        println!("{:?}", v);

        let mut a = &mut *(allocator.alloc::<f32>(10).unwrap() as *mut [f32; 10]);
        for i in 0..10 {
            a[i] = 0.1;
        }
        println!("{:?}", a);
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
