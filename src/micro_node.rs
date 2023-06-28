#[derive(Debug)]
pub struct BLiteNode<'a> {
    pub inputs: &'a [i32],
    pub outputs: &'a [i32],
    // intermidiates: Option<&'a [usize]>,
    // temporaries: Option<&'a [usize]>,
}
