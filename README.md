# BerryLite Micro
[![build](https://github.com/kadu-v/berrylite/actions/workflows/rust.yml/badge.svg?branch=develop)](https://github.com/kadu-v/berrylite/actions/workflows/rust.yml)

BerryLite Micro is the interpreter of TensorFlw model fully implemented  entirely in Rust.
The interpreter is to execute [TensorFlow Lite](https://www.tensorflow.org/lite) models on micro controller, 
and provides APIs similar with [TensorFlow Micro](https://www.tensorflow.org/lite/microcontrollers)'s APIs. 

## How to use
You add the following code to your `Cargo.toml`.
```toml
berrylite = { git = "git@github.com:kadu-v/berrylite.git" }
```

If you want to use this crate on `no_std`, you should enable `no_std` feature.
```toml
berrylite = { git = "git@github.com:kadu-v/berrylite.git", features = ["no_std"] }
```

## Example
This is the `hello_world` example that predicts sin cave. 
If you want to know more examples, you can find other examples in `examples` directory.
```rust
const BUFFER: &[u8; 3164] = include_bytes!("../resources/models/hello_world_float.tflite");
const ARENA_SIZE: usize = 10 * 1024;
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

fn predict(input: f32) -> Result<f32> {
    let model = tflite::root_as_model(BUFFER).unwrap();

    let mut allocator = unsafe { BumpArenaAllocator::new(&mut ARENA) };

    let mut op_resolver = BLiteOpResolver::<1, f32, _>::new();
    op_resolver.add_op(OpFullyConnected::fully_connected())?;

    let mut interpreter = BLiteInterpreter::new(&mut allocator, &op_resolver, &model)?;

  interpreter.input.data[0] = input;
    interpreter.invoke()?;

    let output = interpreter.output;

    Ok(output.data[0])
}
```