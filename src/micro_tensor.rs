use crate::micro_array::{ArrayElem, BLiteArray};
use crate::micro_errors::{BLiteError, Result};
use core::cell::RefCell;

// pub type BLiteTensor<'a, T> = RefCell<BLiteArray<'a, T>>;
pub type BLiteInnerTensor<'a, T> = RefCell<BLiteArray<'a, T>>;

#[derive(Debug)]
pub enum BLiteTensor<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    BTensor(BLiteInnerTensor<'a, T>),
    I32Tensor(BLiteInnerTensor<'a, i32>),
}

impl<'a, T> BLiteTensor<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub fn _t(&self) -> Result<&BLiteInnerTensor<'a, T>> {
        match self {
            BLiteTensor::BTensor(e) => Ok(e),
            BLiteTensor::I32Tensor(_) => Err(BLiteError::NotBTensor),
        }
    }

    pub fn _i32(&self) -> Result<&BLiteInnerTensor<'a, i32>> {
        match self {
            BLiteTensor::BTensor(_) => Err(BLiteError::NotI32Tensor),
            BLiteTensor::I32Tensor(e) => Ok(e),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            BLiteTensor::BTensor(x) => x.borrow().len(),
            BLiteTensor::I32Tensor(x) => x.borrow().len(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            BLiteTensor::BTensor(x) => x.borrow().size(),
            BLiteTensor::I32Tensor(x) => x.borrow().size(),
        }
    }
}
