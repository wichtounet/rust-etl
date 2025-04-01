use std::ops::*;

// Rust is pretty much retarded for getting constants out a generic type
pub trait Constants {
    fn one() -> Self;
}

impl Constants for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Constants for f32 {
    fn one() -> Self {
        1.0
    }
}

impl Constants for i64 {
    fn one() -> Self {
        1
    }
}

impl Constants for i32 {
    fn one() -> Self {
        1
    }
}

pub trait Float {
    fn exp(self) -> Self;
}

impl Float for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
}

impl Float for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
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
            + DivAssign,
    > EtlValueType for T
{
}

#[derive(PartialEq)]
pub enum EtlType {
    Simple,
    Smart,
    Value,
}

pub trait EtlExpr<T: EtlValueType> {
    const DIMENSIONS: usize;
    const TYPE: EtlType;

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

    fn validate_assign<LeftExpr: EtlExpr<T>>(&self, _lhs: &LeftExpr) {
        panic!("This function is only implemented for smart expression");
    }
}

pub trait EtlContainer<T: EtlValueType>: EtlExpr<T> {
    fn get_data(&self) -> &Vec<T>;
}

// It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do. Therefore, I settled on at instead, which should work fine
// TODO: See if there is any way to remove the phantom data here
pub struct EtlWrapper<T: EtlValueType, SubExpr: EtlExpr<T>> {
    pub value: SubExpr,
    pub _marker: std::marker::PhantomData<T>,
}

pub trait EtlComputable<T: EtlValueType> {
    type ComputedAsVector: EtlContainer<T>;
    type ComputedAsMatrix: EtlContainer<T>;
    fn to_vector(&self) -> EtlWrapper<T, Self::ComputedAsVector>;
    fn to_matrix(&self) -> EtlWrapper<T, Self::ComputedAsMatrix>;
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

    if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple {
        if lhs.size() != rhs.size() {
            panic!("Incompatible assignment ([{}] = [{}])", lhs.size(), rhs.size());
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.validate_assign(lhs);
    } else {
        panic!("Unhandled EtlType");
    }
}

pub fn assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    // TODO Ideally, a RightExpr::TYPE = Value should be a simple memcpy

    if RightExpr::DIMENSIONS == 0 {
        // If the other end is a constant, it's not sized and we cannot use rhs.size()
        for i in 0..data.len() {
            data[i] = rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple {
        // I thought we could manually unroll this loop, but it turns out to be slower than the
        // version generated by the compiler

        for i in (0..(rhs.size() + 7) & !7).step_by(1) {
            data[i] = rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into(data);
    } else {
        panic!("Unhandled EtlType");
    }
}

pub fn add_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::DIMENSIONS == 0 {
        // If the other end is a constant, it's not sized and we cannot use rhs.size()
        for i in 0..data.len() {
            data[i] += rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple {
        for i in (0..(rhs.size() + 7) & !7).step_by(1) {
            data[i] += rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_add(data);
    } else {
        panic!("Unhandled EtlType");
    }
}

pub fn sub_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::DIMENSIONS == 0 {
        // If the other end is a constant, it's not sized and we cannot use rhs.size()
        for i in 0..data.len() {
            data[i] -= rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple {
        for i in (0..(rhs.size() + 7) & !7).step_by(1) {
            data[i] -= rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_sub(data);
    } else {
        panic!("Unhandled EtlType");
    }
}

pub fn scale_assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: &RightExpr) {
    if RightExpr::DIMENSIONS == 0 {
        // If the other end is a constant, it's not sized and we cannot use rhs.size()
        for i in 0..data.len() {
            data[i] *= rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Value || RightExpr::TYPE == EtlType::Simple {
        for i in (0..(rhs.size() + 7) & !7).step_by(1) {
            data[i] *= rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into_scale(data);
    } else {
        panic!("Unhandled EtlType");
    }
}
