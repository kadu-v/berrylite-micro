#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TFLiteFusedActivation {
    ActNone,
    ActRelu,
    ActSigmoid,
}

#[derive(Debug)]
pub enum BLiteOpParams {
    FullyConnectedParams {
        activation: TFLiteFusedActivation,
        keep_dims: bool,
    },
    Conv2DParams {
        input_h: i32,
        input_w: i32,
        filter_h: i32,
        filter_w: i32,
        output_h: i32,
        output_w: i32,
    },
}
