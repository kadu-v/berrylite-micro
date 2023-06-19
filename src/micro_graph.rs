#[derive(Debug)]
pub struct TFLiteIntArray<'a> {
    size: usize,
    data: &'a mut [isize],
}

#[derive(Debug)]
pub struct TFLiteFloatArray<'a> {
    size: usize,
    data: &'a mut [f32],
}

#[derive(Debug)]
pub struct TFLiteNode<'a> {
    inputs: &'a mut TFLiteFloatArray<'a>,
    outputs: &'a mut TFLiteFloatArray<'a>,
    intermidiates: &'a mut TFLiteFloatArray<'a>,
    temporaries: &'a mut TFLiteFloatArray<'a>,
}
