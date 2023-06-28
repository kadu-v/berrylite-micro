use core::fmt::Debug;

use crate::micro_array::BLiteArray;

#[derive(Debug)]
pub struct BLiteContext<'a, T: Debug> {
    tensors: &'a [BLiteArray<'a, T>],
}

impl<'a, T: Debug> BLiteContext<'a, T> {
    pub const fn new(
        tensors: &'a [BLiteArray<'a, T>],
    ) -> Self {
        Self { tensors }
    }

    pub fn get_tensors(&self) -> &'a [BLiteArray<'a, T>] {
        self.tensors
    }
}
