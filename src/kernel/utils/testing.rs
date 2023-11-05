use num_traits::FromPrimitive;

use crate::micro_array::ArrayElem;

pub struct Tensor<T>
where
    T: ArrayElem<T>,
{
    pub data: Vec<T>,
    pub shape: Vec<i32>,
}

impl<T> Tensor<T>
where
    T: ArrayElem<T>,
{
    pub fn from(data: Vec<T>, shape: Vec<i32>) -> Self {
        Self { data, shape }
    }

    pub fn ones(shape: &[i32]) -> Self {
        assert!(shape.len() > 0, "expected non empty, but got 0");
        let tot = shape.iter().fold(1usize, |x, &acc| x * acc as usize);
        let v = FromPrimitive::from_usize(1).unwrap();
        let data = vec![v; tot];
        let shape = shape.iter().cloned().collect();
        Self { data, shape }
    }

    pub fn zeros(shape: &[i32]) -> Self {
        assert!(shape.len() > 0, "expected non empty, but got 0");
        let tot = shape.iter().fold(1, |x, &acc| x * acc as usize);
        let v = FromPrimitive::from_usize(0).unwrap();
        let data = vec![v; tot];
        let shape = shape.iter().cloned().collect();
        Self { data, shape }
    }
}
