#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TFLiteFusedActivation {
    ActNone,
    ActRelu,
    ActSigmoid,
}

#[derive(Debug)]
pub enum BuiltinOpData {
    TFLiteFullyConnectedParams {
        activation: TFLiteFusedActivation,
        keep_dims: bool,
    },
}
