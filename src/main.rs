use flatbuffers;
use verdigiris::tflite_schema_generated::tflite;
const BUFFER: &[u8; 3164] = include_bytes!("../models/hello_world_float.tflite");
// const BUFFER_PERSON: &[u8; 300568] = include_bytes!("../models/person_detect.tflite");

fn main() {
    let model = tflite::root_as_model(BUFFER).unwrap();
    let table = model.description();
    println!("{:?}", table);
    let x = model.operator_codes();
    println!("{:?}", x);

    let y = model.subgraphs().unwrap();
    println!("{}", y.len());

    let layer = y.get(0);
    let operators = layer.operators().unwrap();

    for op in operators {
        println!("{:?}", op);
    }

    let tensors = layer.tensors().unwrap();
    for tensor in tensors {
        println!("{:?}", tensor);
    }
}
