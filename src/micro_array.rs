use flatbuffers::Vector;
use num_traits::{AsPrimitive, FromPrimitive};

use crate::micro_allocator::ArenaAllocator;
use crate::micro_errors::{BLiteError::*, Result};
use crate::micro_slice::{alloc_array_mut, from_tflite_vector, from_tflite_vector_mut};
use crate::tflite_schema_generated::tflite::Buffer;
use core::fmt::Debug;
use core::mem::{align_of, size_of};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::slice::from_raw_parts_mut;
use min_max_traits::{Max, Min};

/*-----------------------------------------------------------------------------*/
#[derive(Debug, Clone, Copy)]
pub struct BLiteQuantizationParams<'a> {
    pub scale: &'a [f32],
    pub zero_point: &'a [i64],
}

impl<'a> BLiteQuantizationParams<'a> {
    pub const fn new(scale: &'a [f32], zero_point: &'a [i64]) -> Self {
        Self { scale, zero_point }
    }
}

/*-----------------------------------------------------------------------------*/
pub trait ArrayElem<T: 'static + Clone + Copy> = Debug
    + Clone
    + Copy
    + Add<Output = T>
    + Mul<Output = T>
    + Sub<Output = T>
    + Div<Output = T>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialEq
    + PartialOrd
    + AsPrimitive<f32>
    // TODO: this type should be removed, because it is not used for the quantized runtime.
    // + AsPrimitive<u8>
    + AsPrimitive<i8>
    + AsPrimitive<i32>
    + FromPrimitive
    + Max
    + Min
    + Default;

/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct BLiteArray<'a, T>
where
    T: Debug + Clone + Copy,
{
    pub data: &'a mut [T],
    pub dims: &'a [i32],
    pub quant_params: Option<BLiteQuantizationParams<'a>>,
}

impl<'a, T: ArrayElem<T>> BLiteArray<'a, T> {
    // This method does not initialize the elements of data
    pub unsafe fn new(
        allocator: &mut impl ArenaAllocator,
        data_size: usize,
        dims: &[i32],
        quant_params: Option<BLiteQuantizationParams<'a>>,
    ) -> Result<Self> {
        // TODO: should use check_mul
        let tot_size = dims.iter().fold(1, |x, &acc| x * acc);

        if tot_size != data_size as i32 {
            return Err(NotMatchSize);
        }

        let data_row_ptr = allocator.alloc(size_of::<T>() * data_size, align_of::<T>())?;

        let data = from_raw_parts_mut(data_row_ptr as *mut T, data_size);

        let dims_row_ptr = allocator.alloc(size_of::<usize>() * dims.len(), align_of::<usize>())?;

        // TODO: should not copy the array of dims
        let copied_dims = from_raw_parts_mut(dims_row_ptr as *mut i32, dims.len());

        for (i, &e) in dims.iter().enumerate() {
            copied_dims[i] = e;
        }

        return Ok(Self {
            data,
            dims: copied_dims,
            quant_params,
        });
    }

    pub unsafe fn from_tflite_buffer(
        allocator: &mut impl ArenaAllocator,
        buffer: Buffer<'a>,
        shape: Vector<'a, i32>,
        quant_params: Option<BLiteQuantizationParams<'a>>,
    ) -> Result<Self> {
        if let Some(buffer_data) = buffer.data() {
            let data = from_tflite_vector_mut(&buffer_data);
            let dims = from_tflite_vector(&shape);
            Ok(Self {
                data,
                dims,
                quant_params,
            })
        } else {
            let dims = from_tflite_vector(&shape);
            let data = alloc_array_mut(allocator, 0)?;

            Ok(Self {
                data,
                dims,
                quant_params: quant_params,
            })
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get_quantization_scale_and_zero_point(&self) -> Option<(&'a [f32], &'a [i64])> {
        self.quant_params
            .map(|quant_params| (quant_params.scale, quant_params.zero_point))
    }
}
