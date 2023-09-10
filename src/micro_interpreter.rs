use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteArray};
use crate::micro_erros::Result;
use crate::micro_graph::BLiteGraph;
use crate::micro_op_resolver::BLiteOpResolver;
use crate::tflite_schema_generated::tflite::Model;

#[derive(Debug)]
pub struct BLiteInterpreter<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    version: u32,
    pub input: &'a mut BLiteArray<'a, T>,
    pub output: &'a BLiteArray<'a, T>,
    graph: BLiteGraph<'a, T>,
}

impl<'a, T> BLiteInterpreter<'a, T>
where
    T: ArrayElem<T> + 'a,
{
    pub fn new<const N: usize, S: ArenaAllocator>(
        allocator: &mut S,
        op_resolver: &'a BLiteOpResolver<'a, N, T, S>,
        model: &'a Model<'a>,
    ) -> Result<Self> {
        let version = model.version();

        let graph = BLiteGraph::allocate_graph(allocator, op_resolver, model)?;

        let subgraph = model.subgraphs().unwrap().get(0);
        assert_eq!(
            Self::input_size(model),
            1,
            "Expected the length of input is 1, but got {}",
            Self::input_size(model)
        );
        let input_index = subgraph.inputs().unwrap().get(0) as usize;
        let input = unsafe {
            &mut *((graph.subgraphs[0].borrow().tensors[input_index]._b_tensor()?).as_ptr()
                as *mut BLiteArray<'a, T>)
        };

        assert_eq!(
            Self::output_size(model),
            1,
            "Expected the length of output is 1, but got {}",
            Self::output_size(model)
        );
        let output_index = subgraph.outputs().unwrap().get(0) as usize;
        let output = unsafe {
            &*((graph.subgraphs[0].borrow().tensors[output_index]._b_tensor()?).as_ptr()
                as *const BLiteArray<'a, T>)
        };

        Ok(Self {
            version,
            input,
            output,
            graph,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn invoke(&self) -> Result<()> {
        self.graph.invoke()
    }

    pub fn get_input_quantization_params(&self) -> Option<(f32, i32)> {
        self.input
            .get_quantization_scale_and_zero_point()
            .map(|(scale, zero_point)| (scale[0], zero_point[0] as i32))
    }

    pub fn get_output_quantization_params(&self) -> Option<(f32, i32)> {
        self.output
            .get_quantization_scale_and_zero_point()
            .map(|(scale, zero_point)| (scale[0], zero_point[0] as i32))
    }

    fn input_size(model: &Model<'a>) -> usize {
        model.subgraphs().unwrap().get(0).inputs().unwrap().len()
    }

    fn output_size(model: &Model<'a>) -> usize {
        model.subgraphs().unwrap().get(0).outputs().unwrap().len()
    }
}
