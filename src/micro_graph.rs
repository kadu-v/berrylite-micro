use flatbuffers::{ForwardsUOffset, Vector};

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteArray, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_erros::{BLiteError::*, Result};
use crate::micro_node::BLiteNode;
use crate::micro_op_resolver::BLiteOpResolver;
use crate::micro_registration::BLiteRegistration;
use crate::micro_slice::from_tflite_vector;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::{
    self, Buffer, Model, Operator, OperatorCode, QuantizationParameters,
};

use core::cell::RefCell;
use core::fmt::Debug;
use core::{
    mem::{align_of, size_of},
    slice::from_raw_parts_mut,
};
/*-----------------------------------------------------------------------------*/
/* Type synonyms for TFLiteGraph                                               */
/*-----------------------------------------------------------------------------*/
type TFLiteSubGraph<'a> = tflite::SubGraph<'a>;
type TFLiteOperators<'a> = Vector<'a, ForwardsUOffset<Operator<'a>>>;
type TFLiteOperatorCodes<'a> = Vector<'a, ForwardsUOffset<OperatorCode<'a>>>;
type TFLiteBuffers<'a> = Vector<'a, ForwardsUOffset<Buffer<'a>>>;

/*-----------------------------------------------------------------------------*/
/* Struct for a graph                                                          */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct BLiteGraph<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub subgraphs: &'a [RefCell<BLiteSubgraph<'a, T>>],
}

impl<'a, T> BLiteGraph<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub fn allocate_graph<const N: usize>(
        allocator: &mut impl ArenaAllocator,
        op_resolver: &BLiteOpResolver<N, T>,
        model: &Model<'a>,
    ) -> Result<Self> {
        let Some(subgraphs) = model.subgraphs() else {
            return Err(NotFoundSubgraphs)
        };
        let Some(buffers) = model.buffers() else {
            return Err(NotFoundBuffers)
        };

        let Some(operator_codes) = model.operator_codes() else {
            return Err(NotFoundOperatorCodes);
        };

        assert_eq!(
            subgraphs.len(),
            1,
            "expected the length of subgraphs is 1, but got {}",
            subgraphs.len()
        );
        let blite_subgraphs = unsafe {
            let row_ptr = allocator.alloc(
                subgraphs.len() * size_of::<RefCell<BLiteSubgraph<'a, T>>>(),
                align_of::<RefCell<BLiteSubgraph<'a, T>>>(),
            )?;

            from_raw_parts_mut(
                row_ptr as *mut RefCell<BLiteSubgraph<'a, T>>,
                subgraphs.len(),
            )
        };
        for (i, subgraph) in subgraphs.iter().enumerate() {
            let Some(operators) = subgraph.operators() else {
                return Err(NotFoundOperators)
            };

            let blite_subgraph = BLiteSubgraph::allocate_subgraph(
                allocator,
                &op_resolver,
                &subgraph,
                &operators,
                &operator_codes,
                &buffers,
            )?;
            blite_subgraphs[i] = RefCell::new(blite_subgraph);
        }
        Ok(Self {
            subgraphs: blite_subgraphs,
        })
    }

    pub fn invoke(&self) -> Result<()> {
        for subgraph in self.subgraphs {
            subgraph.borrow_mut().invoke()?;
        }

        Ok(())
    }
}

/*-----------------------------------------------------------------------------*/
/* Struct for a subgraph                                                       */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct BLiteSubgraph<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub node_and_registrations: &'a [(BLiteNode<'a>, BLiteRegistration<T>)],
    pub tensors: &'a mut [BLiteTensor<'a, T>],
}

impl<'a, T> BLiteSubgraph<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub fn new(
        node_and_registrations: &'a [(BLiteNode<'a>, BLiteRegistration<T>)],
        tensors: &'a mut [BLiteTensor<'a, T>],
    ) -> Self {
        Self {
            node_and_registrations,
            tensors,
        }
    }

    pub fn allocate_subgraph<const N: usize>(
        allocator: &mut impl ArenaAllocator,
        op_resolver: &BLiteOpResolver<N, T>,
        subgraph: &TFLiteSubGraph<'a>,
        operators: &TFLiteOperators<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
        buffers: &TFLiteBuffers<'a>,
    ) -> Result<Self> {
        let tensors = Self::allocate_eval_tensors(allocator, subgraph, buffers)?;

        let node_and_registrations = unsafe {
            Self::allocate_node_and_registrations(
                op_resolver,
                allocator,
                operators,
                operator_codes,
                tensors,
            )?
        };

        Ok(Self {
            node_and_registrations,
            tensors,
        })
    }

    fn allocate_eval_tensors(
        allocator: &mut impl ArenaAllocator,
        subgraph: &TFLiteSubGraph<'a>,
        buffers: &TFLiteBuffers<'a>,
    ) -> Result<&'a mut [BLiteTensor<'a, T>]> {
        // size of allocated tensors
        let tensors_size = subgraph.tensors().unwrap().len();

        // Note that tensors
        let tensors = unsafe {
            match allocator.alloc(
                size_of::<BLiteTensor<'a, T>>() * tensors_size,
                align_of::<BLiteTensor<'a, T>>(),
            ) {
                Ok(tensors_row_ptr) => {
                    from_raw_parts_mut(tensors_row_ptr as *mut BLiteTensor<'a, T>, tensors_size)
                }
                Err(err) => return Err(err),
            }
        };

        if let Some(subgraph_tensors) = subgraph.tensors() {
            for (i, tensor) in subgraph_tensors.iter().enumerate() {
                let quant_params = tensor.quantization();
                let blite_quant_params = Self::parse_quant_params(quant_params);
                let tensor_idx = tensor.buffer();
                let buffer = buffers.get(tensor_idx as usize);
                let dims = tensor.shape().unwrap();
                let tflite_tensor = unsafe {
                    BLiteArray::from_tflite_buffer(allocator, buffer, dims, blite_quant_params)?
                };
                tensors[i] = RefCell::new(tflite_tensor);
            }
            Ok(tensors)
        } else {
            Err(NotFoundTensor)
        }
    }

    fn parse_quant_params(
        quant_params: Option<QuantizationParameters<'a>>,
    ) -> Option<BLiteQuantizationParams> {
        if let Some(quant_params) = quant_params {
            let Some(scale) =
                quant_params.scale().map(|x| x.get(0)) else {
                    return None
                };
            let Some(zero_point) = quant_params
                .zero_point()
                .map(|x| x.get(0)) else {
                    return None;
                };

            Some(BLiteQuantizationParams::new(scale, zero_point as i32))
        } else {
            None
        }
    }

    unsafe fn allocate_node_and_registrations<const N: usize>(
        op_resolver: &BLiteOpResolver<N, T>,
        allocator: &mut impl ArenaAllocator,
        operators: &TFLiteOperators<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<&'a [(BLiteNode<'a>, BLiteRegistration<T>)]> {
        let node_and_registrations_row_ptr = allocator.alloc(
            size_of::<(BLiteNode<'_>, BLiteRegistration<T>)>() * operators.len(),
            align_of::<(BLiteNode<'_>, BLiteRegistration<T>)>(),
        )?;
        let node_and_registrations = from_raw_parts_mut(
            node_and_registrations_row_ptr as *mut (BLiteNode<'_>, BLiteRegistration<T>),
            operators.len(),
        );

        for (i, op) in operators.iter().enumerate() {
            let inputs = op.inputs().unwrap();
            let outputs = op.outputs().unwrap();
            let node = Self::allocate_node(&inputs, &outputs)?;
            let registration = Self::alloc_registration(op_resolver, &op, operator_codes, tensors)?;
            node_and_registrations[i] = (node, registration);
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

    unsafe fn alloc_registration<const N: usize>(
        op_resolver: &BLiteOpResolver<N, T>,
        op: &Operator<'a>,
        operator_codes: &TFLiteOperatorCodes<'a>,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteRegistration<T>> {
        let idx = op.opcode_index();
        if idx as usize >= operator_codes.len() {
            return Err(MissingRegistration);
        }

        let op_code = operator_codes.get(idx as usize);
        let builtin_code = op_code.builtin_code();
        let blite_op = op_resolver.find_op(&builtin_code)?;
        let mut registration = blite_op.get_registration();
        let parser = blite_op.get_parser();
        let builtin_option = parser(*op, tensors).unwrap();

        registration.builtin_option = builtin_option;

        return Ok(registration);
    }

    pub fn invoke(&mut self) -> Result<()> {
        let node_and_registrations = self.node_and_registrations;

        let ctx = BLiteContext::new();

        for (_, (node, registration)) in node_and_registrations.iter().enumerate() {
            let tensors = unsafe { &mut *(self.tensors as *mut [BLiteTensor<_>]) };
            let builtin_option = registration.builtin_option;
            let eval = registration.eval;
            eval(&ctx, tensors, node, builtin_option)?;
        }
        Ok(())
    }
}
