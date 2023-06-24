use crate::tflite_schema_generated::tflite::Buffer;

#[derive(Debug)]
pub struct BLiteIntArray<'a> {
    size: usize,
    data: &'a mut [isize],
}

#[derive(Debug)]
pub struct BLiteFloatArray<'a> {
    data: &'a mut [f32],
}

impl<'a> BLiteFloatArray<'a> {
    pub fn new(data: &'a mut [f32]) -> BLiteFloatArray<'a> {
        BLiteFloatArray { data }
    }

    pub fn from_buffer(buffer: Buffer) -> Option<BLiteFloatArray<'a>> {
        if let Some(buffer_data) = buffer.data() {
            let bytes = buffer_data.bytes();
            let data = unsafe {
                core::slice::from_raw_parts_mut(
                    bytes.as_ptr() as *mut f32,
                    bytes.len() / core::mem::size_of::<f32>(),
                )
            };
            return Some(BLiteFloatArray { data: data });
        } else {
            return None;
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a mut BLiteFloatArray<'a>,
    outputs: &'a mut BLiteFloatArray<'a>,
    intermidiates: &'a mut BLiteFloatArray<'a>,
    temporaries: &'a mut BLiteFloatArray<'a>,
}
