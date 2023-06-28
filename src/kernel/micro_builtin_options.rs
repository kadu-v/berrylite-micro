use std::fmt::Debug;

use crate::micro_array::ArrayElem;

#[derive(Debug, Clone, Copy)]
pub enum BLiteBuiltinOption<T: Debug + ArrayElem<T>> {
    FullyConnectedOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
    },
    NotInitialize,
}
