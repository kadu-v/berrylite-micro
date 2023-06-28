use std::fmt::Debug;

use crate::micro_array::ArrayElem;

#[derive(Debug, Clone, Copy)]
pub enum BLiteBuiltinOption<T: Debug + ArrayElem<T>> {
    FullyConnectedOptions { activation: Option<fn(T) -> T> },
    NotInitialize,
}
