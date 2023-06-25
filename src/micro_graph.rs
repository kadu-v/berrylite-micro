use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::BLiteArray;
use crate::micro_ops::Regstration;
use crate::tflite_schema_generated::tflite;
use core::fmt::Debug;
use core::{
    mem::{align_of, size_of, size_of_val},
    slice::from_raw_parts_mut,
};

#[derive(Debug)]
pub struct Subgraph<'a, T: Debug, R: Regstration<'a, T>> {
    node_and_regstrations: &'a [(BLiteNode<'a>, &'a R)],
    tensors: &'a [BLiteArray<'a, T>],
}

impl<'a, T: Debug, R: Regstration<'a, T>>
    Subgraph<'a, T, R>
{
    pub fn new(
        node_and_regstrations: &'a [(
            BLiteNode<'a>,
            &'a R,
        )],
        tensors: &'a [BLiteArray<'a, T>],
    ) -> Self {
        Self {
            node_and_regstrations,
            tensors,
        }
    }

    pub fn allocate_subgraph(
        allocator: &mut impl ArenaAllocator,
        subgraph: &tflite::SubGraph<'_>,
    ) -> Option<Self> {
        // allocate tensors
        let tensors_size =
            subgraph.tensors().unwrap().len();

        println!("ok");
        // Note that tensors
        let mut tensors = unsafe {
            match allocator.alloc(
                size_of::<BLiteArray<'a, T>>()
                    * tensors_size,
                align_of::<BLiteArray<'a, T>>(),
            ) {
                Some(tensors_row_ptr) => {
                    from_raw_parts_mut(
                        tensors_row_ptr
                            as *mut BLiteArray<'a, T>,
                        tensors_size,
                    )
                }
                None => return None,
            }
        };

        for i in 0..tensors_size {
            let tensor: BLiteArray<'_, T> = unsafe {
                BLiteArray::new(allocator, 10, &[10])
                    .unwrap()
            };
            tensors[i] = tensor;
        }

        println!("{:?}", tensors);
        unsafe {
            println!("{}", size_of_val(&tensors));
        }
        return None;
    }

    // fn allocate_eval_tensors(
    //     model: &Model,
    //     subgprah: &mut SubGraph<'a>,
    // ) -> Option<&'a [&'a BLiteArray<'a, T>]> {
    //     todo!()
    // }
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a BLiteArray<'a, usize>,
    outputs: &'a BLiteArray<'a, usize>,
    intermidiates: &'a BLiteArray<'a, usize>,
    temporaries: &'a BLiteArray<'a, usize>,
}
