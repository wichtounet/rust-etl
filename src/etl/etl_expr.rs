use std::ops::*;

pub trait EtlValueType: Default + Clone + Copy + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Mul<Output = Self> + MulAssign {}
impl<T: Default + Clone + Copy + Add<Output = T> + AddAssign<T> + Sub<Output = T> + SubAssign<T> + Mul<Output = T> + MulAssign<T>> EtlValueType for T {}

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
}

// It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do. Therefore, I settled on at instead, which should work fine
// TODO: See if there is any way to remove the phantom data here
pub struct EtlWrapper<T: EtlValueType, SubExpr: EtlExpr<T>> {
    pub value: SubExpr,
    pub _marker: std::marker::PhantomData<T>,
}

pub trait EtlComputable<T: EtlValueType> {
    type ComputedAsVector: EtlExpr<T>;
    type ComputedAsMatrix: EtlExpr<T>;
    fn to_vector(&self) -> EtlWrapper<T, Self::ComputedAsVector>;
    fn to_matrix(&self) -> EtlWrapper<T, Self::ComputedAsMatrix>;
}

pub trait EtlWrappable<T: EtlValueType> {
    type WrappedAs: EtlExpr<T> + EtlComputable<T>;
    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs>;
}

pub trait WrappableExpr<T: EtlValueType>: EtlExpr<T> + EtlWrappable<T> + EtlComputable<T> {}
impl<T: EtlValueType, TT: EtlExpr<T> + EtlWrappable<T> + EtlComputable<T>> WrappableExpr<T> for TT {}

// Assignment functions, probably should be moved elsewhere

pub fn assign_direct<T: EtlValueType, RightExpr: EtlExpr<T>>(data: &mut Vec<T>, rhs: RightExpr) {
    // TODO Validate lhs and rhs

    if RightExpr::TYPE == EtlType::Value {
        /*if lhs.size() != rhs.size() {
            panic!("Cannot assign A=B, dimensions are invalid ({} and {})", lhs.size(), rhs.size());
        }*/

        // TODO: How can we do a memcpy when we do not know RightExpr?
        for i in 0..data.len() {
            data[i] = rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Simple {
        /*if lhs.size() != rhs.size() {
            panic!("Cannot assign A=B, dimensions are invalid ({} and {})", lhs.size(), rhs.size());
        }*/

        for i in 0..data.len() {
            data[i] = rhs.at(i);
        }
    } else if RightExpr::TYPE == EtlType::Smart {
        rhs.compute_into(data);
    } else {
        panic!("Unhandled EtlType");
    }
}
