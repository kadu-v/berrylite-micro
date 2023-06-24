use crate::micro_array::BLiteArray;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteStatus;

#[derive(Debug)]
pub struct Subgraph<'a, T, R: Regstration<'a, T>> {
    node_and_regstrations: &'a [(BLiteNode<'a>, R)],
    tensors: &'a [BLiteArray<'a, T>],
}

impl<'a, T, R: Regstration<'a, T>> Subgraph<'a, T, R> {
    pub fn new(
        node_and_regstrations: &'a [(BLiteNode<'a>, R)],
        tensors: &'a [BLiteArray<'a, T>],
    ) -> Self {
        Self {
            node_and_regstrations,
            tensors,
        }
    }
}

pub trait Regstration<'a, T> {
    fn eval(self, context: BLiteContext, node: &'a BLiteNode<'a>) -> BLiteStatus;
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a BLiteArray<'a, usize>,
    outputs: &'a BLiteArray<'a, usize>,
    intermidiates: &'a BLiteArray<'a, usize>,
    temporaries: &'a BLiteArray<'a, usize>,
}
