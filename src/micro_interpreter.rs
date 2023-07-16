use std::cell::{Ref, RefCell};

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteArray};
use crate::micro_erros::{BLiteError::*, Result};
use crate::micro_graph::{BLiteGraph, BLiteSubgraph};
use crate::micro_op_resolver::BLiteOpResolver;
use crate::tflite_schema_generated::tflite::Model;

#[derive(Debug)]
pub struct BLiteInterpreter<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub input: &'a mut BLiteArray<'a, T>,
    pub output: &'a BLiteArray<'a, T>,
    model: &'a Model<'a>,
    graph: BLiteGraph<'a, T>,
}

impl<'a, T> BLiteInterpreter<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub fn new<const N: usize>(
        allocator: &mut impl ArenaAllocator,
        op_resolver: &BLiteOpResolver<N, T>,
        model: &'a Model<'a>,
    ) -> Result<Self> {
        let graph = BLiteGraph::allocate_graph(
            allocator,
            op_resolver,
            model,
        )?;

        let subgraph = model.subgraphs().unwrap().get(0);
        assert_eq!(
            Self::input_size(model),
            1,
            "Expected the length of input is 1, but got {}",
            Self::input_size(model)
        );
        let input_index =
            subgraph.inputs().unwrap().get(0) as usize;
        let input = unsafe {
            &mut *(graph.subgraphs[0].borrow().tensors
                [input_index]
                .as_ptr()
                as *mut BLiteArray<'a, T>)
        };

        assert_eq!(
            Self::output_size(model),
            1,
            "Expected the length of output is 1, but got {}",
            Self::output_size(model)
        );
        let output_index =
            subgraph.outputs().unwrap().get(0) as usize;
        let output = unsafe {
            &*(graph.subgraphs[0].borrow().tensors
                [output_index]
                .as_ptr()
                as *const BLiteArray<'a, T>)
        };

        Ok(Self {
            input,
            output,
            graph,
            model,
        })
    }

    fn input_size(model: &Model<'a>) -> usize {
        model
            .subgraphs()
            .unwrap()
            .get(0)
            .inputs()
            .unwrap()
            .len()
    }

    fn output_size(model: &Model<'a>) -> usize {
        model
            .subgraphs()
            .unwrap()
            .get(0)
            .outputs()
            .unwrap()
            .len()
    }
    pub fn invoke(&self) -> Result<()> {
        self.graph.invoke()
    }
}
