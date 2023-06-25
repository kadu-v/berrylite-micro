use crate::micro_allocator::ArenaAllocator;
use crate::micro_erros::{BLiteError::*, Result};
use crate::tflite_schema_generated::tflite::Buffer;
use core::fmt::Debug;
use core::mem::{align_of, size_of};
use core::slice::from_raw_parts_mut;

#[derive(Debug)]
pub struct BLiteArray<'a, T: Debug> {
    pub data: &'a mut [T],
    pub dims: &'a [usize],
}

impl<'a, T: Debug> BLiteArray<'a, T> {
    // This method does not initialize the elements of data
    pub unsafe fn new(
        allocator: &mut impl ArenaAllocator,
        data_size: usize,
        dims: &[usize],
    ) -> Result<Self> {
        // TODO: should use chech_mul
        let tot_size =
            dims.iter().fold(1, |x, &acc| x * acc);

        if tot_size != data_size {
            return Err(NotMatchSize);
        }

        let data_row_ptr = allocator.alloc(
            size_of::<T>() * data_size,
            align_of::<T>(),
        )?;
        let data = from_raw_parts_mut(
            data_row_ptr as *mut T,
            data_size,
        );

        let dims_row_ptr = allocator.alloc(
            size_of::<usize>() * dims.len(),
            align_of::<usize>(),
        )?;

        let copied_dims = from_raw_parts_mut(
            dims_row_ptr as *mut usize,
            dims.len(),
        );

        for (i, &e) in dims.iter().enumerate() {
            copied_dims[i] = e;
        }

        return Ok(Self {
            data,
            dims: copied_dims,
        });
    }

    pub fn init(
        data: &'a mut [T],
        dims: &'a [usize],
    ) -> Self {
        Self { data, dims }
    }

    pub unsafe fn from_buffer(
        buffer: Buffer,
        dims: &'a [usize],
    ) -> Option<Self> {
        if let Some(buffer_data) = buffer.data() {
            let bytes = buffer_data.bytes();
            let data = unsafe {
                core::slice::from_raw_parts_mut(
                    bytes.as_ptr() as *mut T,
                    bytes.len() / core::mem::size_of::<T>(),
                )
            };
            return Some(Self { data, dims });
        } else {
            return None;
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug)]
pub enum BLiteDataType {
    Float32,
}
