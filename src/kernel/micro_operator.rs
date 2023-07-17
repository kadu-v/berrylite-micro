pub mod conv2d;
pub mod fully_connected;
pub mod max_pool2d;
pub mod padding;
pub mod reshape;

use crate::builtin_op_data::BLiteOpParams;
use crate::micro_array::ArrayElem;
use crate::micro_erros::Result;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use super::micro_builtin_options::BLiteBuiltinOption;

#[derive(Clone, Copy)]
pub struct BLiteOperator<T>
where
    T: ArrayElem<T>,
{
    registration: BLiteRegistration<T>,
    parser: fn(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>>,
}

impl<T> BLiteOperator<T>
where
    T: ArrayElem<T>,
{
    pub fn get_op_code(&self) -> i32 {
        self.registration.op_code
    }

    pub fn get_registration(&self) -> BLiteRegistration<T> {
        self.registration
    }

    pub fn get_parser(
        &self,
    ) -> fn(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        self.parser
    }
}

impl<T: ArrayElem<T>> Debug for BLiteOperator<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "Operator {{ registration: {:?}, parse:...}}",
            self.registration,
        )?;
        Ok(())
    }
}
