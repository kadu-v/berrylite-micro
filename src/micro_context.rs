use core::fmt::Debug;
use core::marker::PhantomData;

#[derive(Debug)]
pub struct BLiteContext<'a, T>
where
    T: Debug + Clone + Copy,
{
    _x: PhantomData<&'a T>,
}

impl<'a, T> BLiteContext<'a, T>
where
    T: Debug + Clone + Copy,
{
    pub fn new() -> Self {
        Self {
            _x: Default::default(),
        }
    }
}
