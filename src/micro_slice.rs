// This function is used for tflite flatbeffer's vector only
use flatbuffers::Vector;

use core::mem::size_of;
use std::fmt::Debug;

// because of changing lifetime 'b to 'a
pub unsafe fn from_tflite_vector<'b, S, U: Debug>(vector: &Vector<'b, S>) -> &'b [U] {
    let bytes = vector.bytes();
    let data = unsafe {
        core::slice::from_raw_parts(bytes.as_ptr() as *const U, bytes.len() / size_of::<U>())
    };

    return data;
}

// This function is used for tflite flatbeffer's vector only
// because of changing lifetimes 'b to 'a
pub unsafe fn from_tflite_vector_mut<'b, S, U: Debug>(vector: &Vector<'b, S>) -> &'b mut [U] {
    let bytes = vector.bytes();
    let data = unsafe {
        core::slice::from_raw_parts_mut(
            (bytes.as_ptr() as *const U) as *mut U,
            bytes.len() / size_of::<U>(),
        )
    };

    return data;
}
