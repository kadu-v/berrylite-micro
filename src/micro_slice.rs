// This function is used for tflite flatbeffer's vector only
use flatbuffers::Vector;

use crate::micro_allocator::ArenaAllocator;
use crate::micro_erros::Result;
use core::mem::size_of;
use core::{fmt::Debug, mem::align_of};

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

pub unsafe fn alloc_array_mut<'a, T>(
    allocator: &mut impl ArenaAllocator,
    size: usize,
) -> Result<&'a mut [T]> {
    let ptr = allocator.alloc(size_of::<T>() * size, align_of::<T>())?;
    let data = core::slice::from_raw_parts_mut(ptr as *mut T, size);
    Ok(data)
}
