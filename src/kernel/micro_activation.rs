pub mod relu;
use crate::micro_array::ArrayElem;
use relu::relu;

pub fn get_activation<T: ArrayElem<T>>(op_code: i32) -> Option<fn(T) -> T> {
    match op_code {
        1 => Some(relu),
        _ => None,
    }
}
