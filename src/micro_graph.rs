use flatbuffers::{ForwardsUOffset, Vector};

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::BLiteArray;
use crate::micro_erros::{BLiteError::*, Result};
use crate::micro_ops::Regstration;
use crate::tflite_schema_generated::tflite::{
    self, Buffer,
};
use core::fmt::Debug;
use core::{
    mem::{align_of, size_of, size_of_val},
    slice::from_raw_parts_mut,
};

#[derive(Debug)]
pub struct Subgraph<'a, T: Debug, R: Regstration<'a, T>> {
    node_and_regstrations: &'a [(BLiteNode<'a>, &'a R)],
    tensors: &'a mut [BLiteArray<'a, T>],
}

impl<'a, T: Debug, R: Regstration<'a, T>>
    Subgraph<'a, T, R>
{
    pub fn new(
        node_and_regstrations: &'a [(
            BLiteNode<'a>,
            &'a R,
        )],
        tensors: &'a mut [BLiteArray<'a, T>],
    ) -> Self {
        Self {
            node_and_regstrations,
            tensors,
        }
    }

    pub fn allocate_subgraph(
        allocator: &mut impl ArenaAllocator,
        subgraph: &tflite::SubGraph<'a>,
        buffers: &Vector<'_, ForwardsUOffset<Buffer<'_>>>,
    ) -> Result<Self> {
        let tensors = Self::allocate_eval_tensors(
            allocator, subgraph, buffers,
        )?;

        println!("xxxxx {:?}", tensors);
        for (i, tensor) in tensors.iter().enumerate() {
            println!("{}, tensor: {:?}", i, tensor);
        }
        return Err(FailedToCreateGraph);
    }

    fn allocate_eval_tensors(
        allocator: &mut impl ArenaAllocator,
        subgraph: &tflite::SubGraph<'a>,
        buffers: &Vector<'_, ForwardsUOffset<Buffer<'_>>>,
    ) -> Result<&'a mut [BLiteArray<'a, T>]> {
        // size of allocated tensors
        let tensors_size =
            subgraph.tensors().unwrap().len();

        // Note that tensors
        let tensors = unsafe {
            match allocator.alloc(
                size_of::<BLiteArray<'a, T>>()
                    * tensors_size,
                align_of::<BLiteArray<'a, T>>(),
            ) {
                Ok(tensors_row_ptr) => from_raw_parts_mut(
                    tensors_row_ptr
                        as *mut BLiteArray<'a, T>,
                    tensors_size,
                ),
                Err(err) => return Err(err),
            }
        };

        if let Some(subgprah_tensors) = subgraph.tensors() {
            for (i, tensor) in
                subgprah_tensors.iter().enumerate()
            {
                let tensor_idx = tensor.buffer();
                let buffer =
                    buffers.get(tensor_idx as usize);
                let dims = tensor.shape().unwrap();
                let tflite_tensor = unsafe {
                    BLiteArray::from_tflite_buffer(
                        allocator, buffer, dims,
                    )?
                };
                tensors[i] = tflite_tensor;
            }
            return Ok(tensors);
        }
        return Err(NotFoundTensor);
    }

    // fn
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a BLiteArray<'a, usize>,
    outputs: &'a BLiteArray<'a, usize>,
    intermidiates: &'a BLiteArray<'a, usize>,
    temporaries: &'a BLiteArray<'a, usize>,
}
