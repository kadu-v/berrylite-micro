use berrylite::kernel::micro_operator::BLiteOperator;
use berrylite::micro_allocator::BumpArenaAllocator;
use berrylite::micro_graph::*;
use berrylite::micro_op_resolver::BLiteOpResorlver;
use berrylite::tflite_schema_generated::tflite;

const BUFFER: &[u8; 3164] =
    include_bytes!("../models/hello_world_float.tflite");
// const BUFFER: &[u8; 300568] =
// include_bytes!("../models/person_detect.tflite");

// const BUFFER: &[u8; 41240] =
//     include_bytes!("../models/trained_lstm.tflite");

const ARENA_SIZE: usize = 1024 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn main() {
    let model = tflite::root_as_model(BUFFER).unwrap();
    println!("model version: {}", model.version());
    let subgraphs = model.subgraphs().unwrap();
    let subgraph = subgraphs.get(0);
    let buffers = model.buffers().unwrap();

    unsafe {
        let mut allocator =
            BumpArenaAllocator::new(&mut ARENA);

        let mut op_resolver =
            BLiteOpResorlver::<1, f32>::new();
        op_resolver
            .add_op(BLiteOperator::fully_connected());

        let operators = subgraph.operators().unwrap();

        let operator_codes =
            model.operator_codes().unwrap();
        let xsubgraph =
            BLiteSubgraph::<f32>::allocate_subgraph(
                &mut allocator,
                &op_resolver,
                &subgraph,
                &operators,
                &operator_codes,
                &buffers,
            )
            .unwrap();

        for e in xsubgraph.node_and_regstrations {
            println!("{:?}", e);
        }

        for (i, e) in xsubgraph.tensors.iter().enumerate() {
            println!("{} -> {:?}", i, e);
        }
    }
}
