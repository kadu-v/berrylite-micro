use crate::micro_array::BLiteArray;
use core::cell::RefCell;

pub type BLiteTensor<'a, T> = RefCell<BLiteArray<'a, T>>;
