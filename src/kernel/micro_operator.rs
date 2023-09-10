pub mod f32;
pub mod i8;

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::ArrayElem;
use crate::micro_erros::Result;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use super::micro_builtin_options::BLiteBuiltinOption;

#[derive(Clone, Copy)]
pub struct BLiteOperator<'a, T, S>
where
    T: ArrayElem<T>,
    S: ArenaAllocator,
{
    registration: BLiteRegistration<'a, T>,
    parser: fn(
        allocator: &mut S,
        op: Operator,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>>,
}

impl<'a, T, S> BLiteOperator<'a, T, S>
where
    T: ArrayElem<T>,
    S: ArenaAllocator,
{
    pub fn get_op_code(&self) -> i32 {
        self.registration.op_code
    }

    pub fn get_registration(&self) -> BLiteRegistration<'a, T> {
        self.registration
    }

    pub fn get_parser(
        &self,
    ) -> fn(
        allocator: &mut S,
        op: Operator,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>> {
        self.parser
    }
}

impl<'a, T: ArrayElem<T>, S: ArenaAllocator> Debug for BLiteOperator<'a, T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Operator {{ registration: {:?}, parse:...}}",
            self.registration,
        )?;
        Ok(())
    }
}
