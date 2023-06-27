use flatbuffers::{ForwardsUOffset, Vector};

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::BLiteArray;
use crate::micro_erros::{BLiteError::*, Result};
use crate::micro_registration::BLiteRegstration;
use crate::micro_slice::from_tflite_vector;
use crate::tflite_schema_generated::tflite::{
    self, Buffer, Operator, OperatorCode,
};
use core::fmt::Debug;
use core::{
    mem::{align_of, size_of},
    slice::from_raw_parts_mut,
};

type TFLiteSubGraph<'a> = tflite::SubGraph<'a>;
type TFLiteOperators<'a> =
    Vector<'a, ForwardsUOffset<Operator<'a>>>;
type TFLiteOperatorCodes<'a> =
    Vector<'a, ForwardsUOffset<OperatorCode<'a>>>;
type TFLiteBuffers<'a> =
    Vector<'a, ForwardsUOffset<Buffer<'a>>>;

#[derive(Debug)]
pub struct BLiteSubgraph<'a, T: Debug + 'a> {
    pub node_and_regstrations:
        &'a [(BLiteNode<'a>, BLiteRegstration)],
    pub tensors: &'a mut [BLiteArray<'a, T>],
}

impl<'a, T: Debug + 'a> BLiteSubgraph<'a, T> {
    pub fn new(
        node_and_regstrations: &'a [(
            BLiteNode<'a>,
            BLiteRegstration,
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
        subgraph: &TFLiteSubGraph<'a>,
        operators: &TFLiteOperators<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
        buffers: &TFLiteBuffers<'a>,
    ) -> Result<Self> {
        let tensors = Self::allocate_eval_tensors(
            allocator, subgraph, buffers,
        )?;

        let node_and_regstrations = unsafe {
            Self::allocate_node_and_regstrations(
                allocator,
                operators,
                operator_codes,
            )?
        };

        Ok(Self {
            node_and_regstrations,
            tensors,
        })
    }

    fn allocate_eval_tensors(
        allocator: &mut impl ArenaAllocator,
        subgraph: &TFLiteSubGraph<'a>,
        buffers: &TFLiteBuffers<'a>,
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
            Ok(tensors)
        } else {
            Err(NotFoundTensor)
        }
    }

    unsafe fn allocate_node_and_regstrations(
        allocator: &mut impl ArenaAllocator,
        operators: &TFLiteOperators<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
    ) -> Result<&'a [(BLiteNode<'a>, BLiteRegstration)]>
    {
        let node_and_registrations_row_ptr = allocator
            .alloc(
                size_of::<(BLiteNode<'_>, BLiteRegstration)>()
                    * operators.len(),
                align_of::<(BLiteNode<'_>, BLiteRegstration)>(),
            )?;
        let node_and_registrations = from_raw_parts_mut(
            node_and_registrations_row_ptr
                as *mut (BLiteNode<'_>, BLiteRegstration),
            operators.len(),
        );

        for (i, op) in operators.iter().enumerate() {
            let inputs = op.inputs().unwrap();
            let outputs = op.outputs().unwrap();
            let node =
                Self::allocate_node(&inputs, &outputs)?;
            let regstration = Self::alloc_regstration(
                operators,
                operator_codes,
            )?;
            node_and_registrations[i] = (node, regstration);
        }

        Ok(node_and_registrations)
    }

    unsafe fn allocate_node(
        inputs: &Vector<'a, i32>,
        outputs: &Vector<'a, i32>,
    ) -> Result<BLiteNode<'a>> {
        let node_inputs = from_tflite_vector(&inputs);
        let node_outputs = from_tflite_vector(&outputs);
        Ok(BLiteNode {
            inputs: node_inputs,
            outputs: node_outputs,
        })
    }

    unsafe fn alloc_regstration(
        operators: &TFLiteOperators<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
    ) -> Result<BLiteRegstration> {
        for (i, op) in operators.iter().enumerate() {
            let idx = op.opcode_index();
            if idx as usize >= operator_codes.len() {
                return Err(MissingRegstration);
            }

            let op_code = operator_codes.get(idx as usize);
            let builtin_code = op_code.builtin_code();
        }
        Ok(BLiteRegstration::default())
    }

    fn get_regstration_from_op_code(
    ) -> Result<BLiteRegstration> {
        todo!()
    }
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a [i32],
    outputs: &'a [i32],
    // intermidiates: Option<&'a [usize]>,
    // temporaries: Option<&'a [usize]>,
}
