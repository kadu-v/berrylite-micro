use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::*;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnectedInt8 {}

impl OpFullyConnectedInt8 {
    const OPCODE: i32 = 9;

    pub fn fully_connected_int8<T: ArrayElem<T>>(
    ) -> BLiteOperator<T> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        let builtin_option =
            op.builtin_options_as_fully_connected_options();
        let mut op_code = -1;
        if let Some(builtin_option) = builtin_option {
            op_code = builtin_option
                .fused_activation_function()
                .0 as i32;
        }

        let input_idx =
            op.inputs().unwrap().get(0) as usize;
        let input_offset = -tensors[input_idx]
            .borrow()
            .quant_params
            .unwrap()
            .zero_point;

        let filter_idx =
            op.inputs().unwrap().get(1) as usize;
        let filter_offset = -tensors[filter_idx]
            .borrow()
            .quant_params
            .unwrap()
            .zero_point;

        let output_idx =
            op.outputs().unwrap().get(0) as usize;
        let output_offset = tensors[output_idx]
            .borrow()
            .quant_params
            .unwrap()
            .zero_point;
        // let output_multiplier =
        //     Self::get_quatized_convolutional_multiplier(
        //         input_scale,
        //         filte_scale,
        //         output_scale,
        //         bias_scale,
        //     );
        let activation = get_activation::<T>(op_code);
        todo!()
        // Ok(BLiteBuiltinOption::FullyConnectedInt8Options {
        //     op_code,
        //     activation,
        // })
    }

    // FullyConnectedParamsQuantized: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected_common.cc#L34
    // OpDataFullyConnected: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected.h#L26
    // CalculateOpDataFullyConnected: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected_common.cc#L62-L63
    fn get_quatized_convolutional_multiplier(
        input_scale: f32,
        filte_scale: f32,
        output_scale: f32,
        bias_scale: Option<f32>,
    ) -> Result<f32> {
        let input_product_scale = input_scale * filte_scale;
        if let Some(bias_scale) = bias_scale {
            let scale_diff =
                (input_product_scale - bias_scale).abs();
            if scale_diff / output_scale <= 0.02 {
                return Err(NotMatchScale(
                    scale_diff / output_scale,
                ));
            }
        }

        let multiplier = input_product_scale / output_scale;
        return Ok(multiplier);
    }

    pub fn registration<T: ArrayElem<T>>(
    ) -> BLiteRegistration<T> {
        BLiteRegistration::new(
            Self::OPCODE,
            Self::eval::<T>,
            NotInitialize,
        )
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input].borrow();

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter].borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();

        let activation = match builtin_option {
            FullyConnectedOptions {
                op_code: _,
                activation,
            } => activation,
            NotInitialize => {
                return Err(NotInitializeActivation)
            }
            _ => return Err(NotCompatibleOption),
        };

        // TODO:
        let batches = 1usize;
        let output_depth =
            filter.dims[filter.dims.len() - 2] as usize;
        let accum_depth =
            filter.dims[filter.dims.len() - 1] as usize;

        for batch in 0..batches {
            for out_d in 0..output_depth {
                let mut total: T = Default::default();
                for acc_d in 0..accum_depth {
                    total += input.data
                        [batch * accum_depth + acc_d]
                        * filter.data
                            [out_d * accum_depth + acc_d];
                }
                output.data[batch * output_depth + out_d] =
                    total;

                let idx_bias = node.inputs[2];
                if idx_bias >= 0 {
                    let bias =
                        tensors[idx_bias as usize].borrow();
                    output.data
                        [batch * output_depth + out_d] +=
                        bias.data[out_d];
                }
            }
        }
        if let Some(activation) = activation {
            for i in 0..output.data.len() {
                output.data[i] = activation(output.data[i]);
            }
        }

        Ok(())
    }
}
