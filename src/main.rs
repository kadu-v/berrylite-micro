use flatbuffers;
use verdigiris::tflite_schema_generated::tflite;
const BUFFER: &[u8; 3164] = include_bytes!("../models/hello_world_float.tflite");

fn main() {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let metadata = model.metadata().unwrap();

    for e in metadata {
        println!("{:?}", e);
    }
}
