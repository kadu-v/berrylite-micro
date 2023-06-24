#[derive(Debug)]
pub struct BLiteIntArray<'a> {
    size: usize,
    data: &'a mut [isize],
}

#[derive(Debug)]
pub struct BLiteFloatArray<'a> {
    size: usize,
    data: &'a mut [f32],
}

#[derive(Debug)]
pub struct BLiteNode<'a> {
    inputs: &'a mut BLiteFloatArray<'a>,
    outputs: &'a mut BLiteFloatArray<'a>,
    intermidiates: &'a mut BLiteFloatArray<'a>,
    temporaries: &'a mut BLiteFloatArray<'a>,
}
