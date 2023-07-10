use crate::micro_array::ArrayElem;

pub fn relu<T: ArrayElem<T>>(x: T) -> T {
    let zero = Default::default();
    if x < zero {
        return zero;
    } else {
        x
    }
}
