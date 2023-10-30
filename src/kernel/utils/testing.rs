use num_traits::{AsPrimitive, FromPrimitive};

use crate::micro_array::ArrayElem;

pub struct Tensor<T>
where
    T: ArrayElem<T>,
{
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: ArrayElem<T>,
{
    pub fn from(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn ones(shape: &[usize]) -> Self {
        assert!(shape.len() > 0, "expected non empty, but got 0");
        let tot = shape.iter().fold(1usize, |x, &acc| x * acc);
        let v = FromPrimitive::from_usize(1).unwrap();
        let data = vec![v; tot];
        let shape = shape.iter().cloned().collect();
        Self { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        assert!(shape.len() > 0, "expected non empty, but got 0");
        let tot = shape.iter().fold(1, |x, &acc| x * acc);
        let v = FromPrimitive::from_usize(0).unwrap();
        let data = vec![v; tot];
        let shape = shape.iter().cloned().collect();
        Self { data, shape }
    }
}
