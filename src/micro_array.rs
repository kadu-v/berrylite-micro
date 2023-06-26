use flatbuffers::Vector;

use crate::micro_allocator::ArenaAllocator;
use crate::micro_erros::{BLiteError::*, Result};
use crate::tflite_schema_generated::tflite::Buffer;
use core::fmt::Debug;
use core::mem::{align_of, size_of};
use core::slice::from_raw_parts_mut;

#[derive(Debug)]
pub struct BLiteArray<'a, T: Debug> {
    pub data: &'a mut [T],
    pub dims: &'a [i32],
}

impl<'a, T: Debug> BLiteArray<'a, T> {
    // This method does not initialize the elements of data
    pub unsafe fn new(
        allocator: &mut impl ArenaAllocator,
        data_size: usize,
        dims: &[i32],
    ) -> Result<Self> {
        // TODO: should use chech_mul
        let tot_size =
            dims.iter().fold(1, |x, &acc| x * acc);

        if tot_size != data_size as i32 {
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
            dims_row_ptr as *mut i32,
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
        dims: &'a [i32],
    ) -> Self {
        Self { data, dims }
    }

    pub unsafe fn from_tflite_buffer(
        allocator: &mut impl ArenaAllocator,
        buffer: Buffer,
        shape: Vector<'a, i32>,
    ) -> Result<Self> {
        if let Some(buffer_data) = buffer.data() {
            let data =
                Self::from_tflite_vector_mut(buffer_data);
            let dims = Self::from_tflite_vector(shape);
            Ok(Self { data, dims })
        } else {
            let data_size = shape
                .iter()
                .fold(1usize, |x, acc| x * acc as usize);
            let dims = Self::from_tflite_vector(shape);
            Self::new(allocator, data_size, dims)
        }
    }

    // This functuion is used for tflite flatbeffer's vector only
    // because of chainging lifetims 'b to 'a
    unsafe fn from_tflite_vector<'b, S, U>(
        vector: Vector<'b, S>,
    ) -> &'a [U] {
        let bytes = vector.bytes();
        let data = unsafe {
            core::slice::from_raw_parts(
                bytes.as_ptr() as *const U,
                bytes.len() / core::mem::size_of::<U>(),
            )
        };
        return data;
    }

    // This functuion is used for tflite flatbeffer's vector only
    // because of chainging lifetims 'b to 'a
    unsafe fn from_tflite_vector_mut<'b, S, U>(
        vector: Vector<'b, S>,
    ) -> &'a mut [U] {
        let bytes = vector.bytes();
        let data = unsafe {
            core::slice::from_raw_parts_mut(
                bytes.as_ptr() as *mut U,
                bytes.len() / core::mem::size_of::<U>(),
            )
        };
        return data;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
