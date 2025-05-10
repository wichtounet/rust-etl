use rayon;
use std::ops::*;

use crate::base_traits::Constants;

pub fn padded_size(size: usize) -> usize {
    (size + 7) & !7
}

pub trait EtlValueType:
    Constants
    + Default
    + Clone
    + Copy
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + std::marker::Sync
    + std::marker::Send
    + std::fmt::Display
    + std::simd::SimdElement
{
}

impl<
        T: Constants
            + Default
            + Clone
            + Copy
            + PartialOrd
            + Neg<Output = T>
            + Add<Output = T>
            + AddAssign<T>
            + Sub<Output = T>
            + SubAssign<T>
            + Mul<Output = T>
            + MulAssign<T>
            + Div<Output = Self>
            + DivAssign
            + std::marker::Sync
            + std::marker::Send
            + std::fmt::Display
            + std::simd::SimdElement,
    > EtlValueType for T
{
}

#[derive(PartialEq, Eq)]
pub enum EtlType {
    Simple,
    Unaligned,
    Smart,
    Value,
}

// Since PartialEq is not const fn (yet), we must declare a const fn comparison
pub const fn is_same_type(lhs_type: EtlType, rhs_type: EtlType) -> bool {
    match (lhs_type, rhs_type) {
        (EtlType::Simple, EtlType::Simple) => true,
        (EtlType::Unaligned, EtlType::Unaligned) => true,
        (EtlType::Smart, EtlType::Smart) => true,
        (EtlType::Value, EtlType::Value) => true,
        _ => false,
    }
}

pub const fn simple_unary_type(etl_type: EtlType) -> EtlType {
    if is_same_type(etl_type, EtlType::Unaligned) {
        EtlType::Unaligned
    } else {
        EtlType::Simple
    }
}

pub const fn simple_binary_type(lhs_type: EtlType, rhs_type: EtlType) -> EtlType {
    if is_same_type(lhs_type, EtlType::Unaligned) || is_same_type(rhs_type, EtlType::Unaligned) {
        EtlType::Unaligned
    } else {
        EtlType::Simple
    }
}

impl EtlType {
    pub fn direct(self) -> bool {
        self == EtlType::Value || self == EtlType::Smart
    }
}

pub trait EtlExpr<T: EtlValueType>: std::marker::Sync {
    const DIMENSIONS: usize;
    const TYPE: EtlType;
    const THREAD_SAFE: bool;

    type Iter<'x>: Iterator<Item = T>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_>;

    /// Return the size of the Expressions.
    ///
    /// This is valid for all dimensions
    fn size(&self) -> usize;

    /// Return the element at position `i`
    ///
    /// This works for all dimensions and consider a flat structure
    fn at(&self, i: usize) -> T;

    fn rows(&self) -> usize;

    // TODO I should find a solution to prevent this at compile-time (without combinatorial explosion)

    fn columns(&self) -> usize {
        panic!("This function is only implemented for 2D containers");
    }

    fn at2(&self, _row: usize, _column: usize) -> T {
        panic!("This function is only implemented for 2D containers");
    }

    fn compute_into(&self, _lhs: &mut Vec<T>) {
        panic!("This function is only implemented for smart expression");
    }

    fn compute_into_add(&self, _lhs: &mut Vec<T>) {
        panic!("This function is only implemented for smart expression");
    }

    fn compute_into_sub(&self, _lhs: &mut Vec<T>) {
        panic!("This function is only implemented for smart expression");
    }

    fn compute_into_scale(&self, _lhs: &mut Vec<T>) {
        panic!("This function is only implemented for smart expression");
    }

    fn compute_into_div(&self, _lhs: &mut Vec<T>) {
        panic!("This function is only implemented for smart expression");
    }

    fn validate_assign<LeftExpr: EtlExpr<T>>(&self, _lhs: &LeftExpr) {
        panic!("This function is only implemented for smart expression");
    }

    fn get_data(&self) -> &Vec<T> {
        panic!("This function is only implemented for direct expression");
    }
}

// It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do. Therefore, I settled on at instead, which should work fine
// TODO: See if there is any way to remove the phantom data here
pub struct EtlWrapper<T: EtlValueType, SubExpr: EtlExpr<T>> {
    pub value: SubExpr,
    pub _marker: std::marker::PhantomData<T>,
}

pub trait EtlComputable<T: EtlValueType> {
    fn to_data(&self) -> Vec<T>;
}

pub trait EtlWrappable<T: EtlValueType> {
    type WrappedAs: EtlExpr<T> + EtlComputable<T>;
    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs>;
}

pub trait WrappableExpr<T: EtlValueType>: EtlExpr<T> + EtlWrappable<T> + EtlComputable<T> {}
impl<T: EtlValueType, TT: EtlExpr<T> + EtlWrappable<T> + EtlComputable<T>> WrappableExpr<T> for TT {}

// Assignment functions, probably should be moved elsewhere

pub fn validate_assign<T: EtlValueType, LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>>(lhs: &LeftExpr, rhs: &RightExpr) {
    if RightExpr::DIMENSIONS == 0 {
        return;
    }

    if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple || RightExpr::TYPE == EtlType::Unaligned {
        if lhs.size() != rhs.size() {
            panic!("Incompatible assignment ([{}] = [{}])", lhs.size(), rhs.size());
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.validate_assign(lhs);
    } else {
        panic!("Unhandled EtlType");
    }
}

// We must use a scope for threads because our data does not have ´static lifetime which would be
// required otherwise
// Ideally, we would uses std::thread::scope, but it does not use a thread pool
// rayon setups the thread pool with max number of CPUs on the machine, so we use that as our
// threads

// Currently, rayon has a massive overhead (when compared to ETL)
// So, we must use a rather high threshold
const PARALLEL_THRESHOLD: usize = 256 * 1024;

pub fn assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    // TODO Ideally, a RightExpr::TYPE = Value should be a simple memcpy

    if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into(data);
    } else {
        let size = if RightExpr::DIMENSIONS == 0 {
            data.len()
        } else if RightExpr::TYPE != EtlType::Unaligned {
            padded_size(rhs.size())
        } else {
            rhs.size()
        };

        if RightExpr::THREAD_SAFE && size > PARALLEL_THRESHOLD {
            rayon::scope(|s| {
                let n = rayon::current_num_threads();
                let block_size = size / n;

                let ptr = data.as_mut_ptr();

                for t in 0..n {
                    let start = t * block_size;
                    let end = if t < n - 1 { (t + 1) * block_size } else { size };

                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), end - start) };

                    s.spawn(move |_| {
                        for (i, x) in slice.iter_mut().enumerate() {
                            *x = rhs.at(start + i);
                        }
                    });
                }
            });
        } else {
            // I thought we could manually unroll this loop (since we can unroll safely by the
            // forced alignment), but it turns out to be slower than the
            // version generated by the compiler

            for (lhs, rhs) in data[0..size].iter_mut().zip(rhs.iter()) {
                *lhs = rhs;
            }
        }
    }
}

pub fn add_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_add(data);
    } else {
        let size = if RightExpr::DIMENSIONS == 0 {
            data.len()
        } else if RightExpr::TYPE != EtlType::Unaligned {
            padded_size(rhs.size())
        } else {
            rhs.size()
        };

        if RightExpr::THREAD_SAFE && size > PARALLEL_THRESHOLD {
            rayon::scope(|s| {
                let n = rayon::current_num_threads();
                let block_size = size / n;

                let ptr = data.as_mut_ptr();

                for t in 0..n {
                    let start = t * block_size;
                    let end = if t < n - 1 { (t + 1) * block_size } else { size };

                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), end - start) };

                    s.spawn(move |_| {
                        for (i, x) in slice.iter_mut().enumerate() {
                            *x += rhs.at(start + i);
                        }
                    });
                }
            });
        } else {
            for (lhs, rhs) in data[0..size].iter_mut().zip(rhs.iter()) {
                *lhs += rhs;
            }
        }
    }
}

pub fn sub_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_sub(data);
    } else {
        let size = if RightExpr::DIMENSIONS == 0 {
            data.len()
        } else if RightExpr::TYPE != EtlType::Unaligned {
            padded_size(rhs.size())
        } else {
            rhs.size()
        };

        if RightExpr::THREAD_SAFE && size > PARALLEL_THRESHOLD {
            rayon::scope(|s| {
                let n = rayon::current_num_threads();
                let block_size = size / n;

                let ptr = data.as_mut_ptr();

                for t in 0..n {
                    let start = t * block_size;
                    let end = if t < n - 1 { (t + 1) * block_size } else { size };

                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), end - start) };

                    s.spawn(move |_| {
                        for (i, x) in slice.iter_mut().enumerate() {
                            *x -= rhs.at(start + i);
                        }
                    });
                }
            });
        } else {
            for (lhs, rhs) in data[0..size].iter_mut().zip(rhs.iter()) {
                *lhs -= rhs;
            }
        }
    }
}

pub fn div_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_div(data);
    } else {
        let size = if RightExpr::DIMENSIONS == 0 {
            data.len()
        } else if RightExpr::TYPE != EtlType::Unaligned {
            padded_size(rhs.size())
        } else {
            rhs.size()
        };

        if RightExpr::THREAD_SAFE && size > PARALLEL_THRESHOLD {
            rayon::scope(|s| {
                let n = rayon::current_num_threads();
                let block_size = size / n;

                let ptr = data.as_mut_ptr();

                for t in 0..n {
                    let start = t * block_size;
                    let end = if t < n - 1 { (t + 1) * block_size } else { size };

                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), end - start) };

                    s.spawn(move |_| {
                        for (i, x) in slice.iter_mut().enumerate() {
                            *x /= rhs.at(start + i);
                        }
                    });
                }
            });
        } else {
            for (lhs, rhs) in data[0..size].iter_mut().zip(rhs.iter()) {
                *lhs /= rhs;
            }
        }
    }
}

pub fn scale_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_scale(data);
    } else {
        let size = if RightExpr::DIMENSIONS == 0 {
            data.len()
        } else if RightExpr::TYPE != EtlType::Unaligned {
            padded_size(rhs.size())
        } else {
            rhs.size()
        };

        if RightExpr::THREAD_SAFE && size > PARALLEL_THRESHOLD {
            rayon::scope(|s| {
                let n = rayon::current_num_threads();
                let block_size = size / n;

                let ptr = data.as_mut_ptr();

                for t in 0..n {
                    let start = t * block_size;
                    let end = if t < n - 1 { (t + 1) * block_size } else { size };

                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), end - start) };

                    s.spawn(move |_| {
                        for (i, x) in slice.iter_mut().enumerate() {
                            *x *= rhs.at(start + i);
                        }
                    });
                }
            });
        } else {
            for (lhs, rhs) in data[0..size].iter_mut().zip(rhs.iter()) {
                *lhs *= rhs;
            }
        }
    }
}

// Parallel dispatchers

pub fn dispatch_parallel_2d<T: EtlValueType, F: Fn(&mut [T], usize, usize) + Sync + Send + Clone>(
    data: &mut Vec<T>,
    size: usize,
    helper: bool,
    mul: usize,
    functor: F,
) {
    if size >= rayon::current_num_threads() && helper {
        rayon::scope(|s| {
            let n = rayon::current_num_threads();
            let block_size = size / n;

            let ptr = data.as_mut_ptr();

            for t in 0..n {
                let start = t * block_size;
                let end = if t < n - 1 { (t + 1) * block_size } else { size };

                let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(start * mul), (end - start) * mul) };

                let f = functor.clone();

                s.spawn(move |_| {
                    f(slice, start, end);
                });
            }
        });
    } else {
        functor(data, 0, size);
    }
}

pub fn dispatch_parallel_block<T: EtlValueType, F: Fn(&mut [T], usize, usize) + Sync + Send + Clone>(data: &mut Vec<T>, size: usize, block: usize, functor: F) {
    if size >= block * 2 {
        rayon::scope(|s| {
            let mut n = rayon::current_num_threads();
            let mut block_size = size / n;

            if block_size < block {
                block_size = block;
                n = size / block_size;
            }

            let ptr = data.as_mut_ptr();

            for t in 0..n {
                let start = t * block_size;
                let end = if t < n - 1 { (t + 1) * block_size } else { size };

                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };

                let f = functor.clone();

                s.spawn(move |_| {
                    f(slice, start, end);
                });
            }
        });
    } else {
        functor(data, 0, size);
    }
}

// Helpers

pub fn forward_data_binary<
    T: EtlValueType,
    F: Fn(&mut Vec<T>, &Vec<T>, &Vec<T>),
    LeftExpr: EtlComputable<T> + EtlExpr<T>,
    RightExpr: EtlComputable<T> + EtlExpr<T>,
>(
    output: &mut Vec<T>,
    lhs: &LeftExpr,
    rhs: &RightExpr,
    functor: F,
) {
    if LeftExpr::TYPE.direct() && RightExpr::TYPE.direct() {
        functor(output, lhs.get_data(), rhs.get_data());
    } else if LeftExpr::TYPE.direct() && !RightExpr::TYPE.direct() {
        let rhs_data = rhs.to_data();

        functor(output, lhs.get_data(), &rhs_data);
    } else if !LeftExpr::TYPE.direct() && RightExpr::TYPE.direct() {
        let lhs_data = lhs.to_data();

        functor(output, &lhs_data, rhs.get_data());
    } else {
        let lhs_data = lhs.to_data();
        let rhs_data = rhs.to_data();

        functor(output, &lhs_data, &rhs_data);
    }
}

pub fn forward_data_binary_mut<
    T: EtlValueType,
    F: FnMut(&mut Vec<T>, &Vec<T>, &Vec<T>),
    LeftExpr: EtlComputable<T> + EtlExpr<T>,
    RightExpr: EtlComputable<T> + EtlExpr<T>,
>(
    output: &mut Vec<T>,
    lhs: &LeftExpr,
    rhs: &RightExpr,
    functor: &mut F,
) {
    if LeftExpr::TYPE.direct() && RightExpr::TYPE.direct() {
        functor(output, lhs.get_data(), rhs.get_data());
    } else if LeftExpr::TYPE.direct() && !RightExpr::TYPE.direct() {
        let rhs_data = rhs.to_data();

        functor(output, lhs.get_data(), &rhs_data);
    } else if !LeftExpr::TYPE.direct() && RightExpr::TYPE.direct() {
        let lhs_data = lhs.to_data();

        functor(output, &lhs_data, rhs.get_data());
    } else {
        let lhs_data = lhs.to_data();
        let rhs_data = rhs.to_data();

        functor(output, &lhs_data, &rhs_data);
    }
}

pub fn forward_data_unary<T: EtlValueType, F: Fn(&mut Vec<T>, &Vec<T>), Expr: EtlComputable<T> + EtlExpr<T>>(output: &mut Vec<T>, expr: &Expr, functor: F) {
    if Expr::TYPE.direct() {
        functor(output, expr.get_data());
    } else {
        let data = expr.to_data();
        functor(output, &data);
    }
}

pub fn forward_data_unary_mut<T: EtlValueType, F: FnMut(&mut Vec<T>, &Vec<T>), Expr: EtlComputable<T> + EtlExpr<T>>(
    output: &mut Vec<T>,
    expr: &Expr,
    functor: &mut F,
) {
    if Expr::TYPE.direct() {
        functor(output, expr.get_data());
    } else {
        let data = expr.to_data();
        functor(output, &data);
    }
}
