// use core::fmt::Debug;
// use core::mem::size_of;
// use core::ops::BitAnd;
// use num_traits::FromPrimitive;
// use std::cmp::{Ord, PartialOrd};
// use std::ops::{Add, Div, Mul, Not, Shr, Sub};

// pub trait FixedPointTrait<T> = Debug
//     + Clone
//     + Copy
//     + PartialEq
//     + Eq
//     + PartialOrd
//     + Ord
//     + FromPrimitive
//     + Add<Output = T>
//     + Mul<Output = T>
//     + Sub<Output = T>
//     + Div<Output = T>
//     + BitAnd<Output = T>
//     + Not<Output = T>
//     + Shr<Output = T>;

// pub struct FixedPoint<T>
// where
//     T: FixedPointTrait<T>,
// {
//     pub total_bits: i32,
//     pub integer_bits: i32,
//     pub fraction_bits: i32,
//     raw_val: T,
// }

// impl<T> FixedPoint<T>
// where
//     T: FixedPointTrait<T>,
// {
//     pub const fn new(x: T, n: i32) -> Self {
//         Self {
//             total_bits: 8 * size_of::<T>() as i32,
//             integer_bits: n,
//             fraction_bits: 8 * size_of::<T>() as i32 - 1 - n,
//             raw_val: x,
//         }
//     }

//     pub fn rescale(&self, dst_integer_bits: i32) -> FixedPoint<T> {
//         let exponent = self.integer_bits - dst_integer_bits;
//         let x = Self::saturating_rounding_muliply_by_pot(self.raw_val, exponent);
//         let result = FixedPoint::new(x, dst_integer_bits);
//         return result;
//     }

//     pub fn saturating_rounding_muliply_by_pot(x: T, exponent: i32) -> T {
//         Self::rounding_divide_by_pot(x, -exponent)
//     }

//     fn rounding_divide_by_pot(x: T, exponent: i32) -> T {
//         // have to implement Dup
//         let mask: T = FromPrimitive::from_i64(((1 as i64) << exponent) - 1).unwrap();
//         let zero: T = FromPrimitive::from_i64(0i64).unwrap();
//         let one: T = FromPrimitive::from_i64(1i64).unwrap();
//         let exp: T = FromPrimitive::from_i32(exponent).unwrap();
//         let remainder = x & mask;
//         let threshold = (mask >> one) + (Self::mask_if_less_than(x, zero) & one);
//         let result = (x >> exp) + (Self::mask_greater_than(remainder, threshold) & one);
//         return result;
//     }

//     fn mask_if_less_than(x: T, y: T) -> T {
//         Self::mask_if_non_zero(x < y)
//     }

//     fn mask_greater_than(x: T, y: T) -> T {
//         Self::mask_if_non_zero(x > y)
//     }

//     fn mask_if_non_zero(b: bool) -> T {
//         let zero: T = FromPrimitive::from_i64(0).unwrap();
//         if b {
//             !zero
//         } else {
//             zero
//         }
//     }
// }
