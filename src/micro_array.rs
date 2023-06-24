use crate::tflite_schema_generated::tflite::Buffer;

#[derive(Debug)]
pub struct BLiteArray<'a, T> {
    pub data: &'a mut [T],
    pub dims: &'a [usize],
}

impl<'a, T> BLiteArray<'a, T> {
    pub fn new(data: &'a mut [T], dims: &'a [usize]) -> Self {
        Self { data, dims }
    }

    pub unsafe fn from_buffer(buffer: Buffer, dims: &'a [usize]) -> Option<Self> {
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
