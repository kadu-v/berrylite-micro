use crate::micro_allocator::ArenaAllocator;
use crate::tflite_schema_generated::tflite::Buffer;
use core::mem::{align_of, size_of};
use core::slice::from_raw_parts_mut;

#[derive(Debug)]
pub struct BLiteArray<'a, T> {
    pub data: &'a mut [T],
    pub dims: &'a [usize],
}

impl<'a, T> BLiteArray<'a, T> {
    // This method does not initialize the elements of data
    pub unsafe fn new(
        allocator: &mut impl ArenaAllocator,
        data_size: usize,
        dims: &[usize],
    ) -> Option<Self> {
        let tot_size = dims
            .iter()
            .fold(Some(1), |x, &acc| x.map(|e| e * acc));

        if let Some(tot_size) = tot_size {
            if tot_size != data_size {
                return None;
            }
        } else {
            return None;
        }

        let data_row_ptr = match allocator.alloc(
            size_of::<T>() * data_size,
            align_of::<T>(),
        ) {
            Some(data_row_ptr) => data_row_ptr,
            None => return None,
        };
        let data = from_raw_parts_mut(
            data_row_ptr as *mut T,
            data_size,
        );

        let dims_row_ptr = match allocator.alloc(
            size_of::<usize>() * dims.len(),
            align_of::<usize>(),
        ) {
            Some(dims_row_ptr) => dims_row_ptr,
            None => return None,
        };
        let copied_dims = from_raw_parts_mut(
            dims_row_ptr as *mut usize,
            dims.len(),
        );

        for (i, &e) in dims.iter().enumerate() {
            copied_dims[i] = e;
        }

        return Some(Self {
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
