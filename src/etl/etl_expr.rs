use std::ops::*;

pub trait EtlValueType: Default + Clone + Copy + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign + Mul<Output = Self> + MulAssign {}
impl<T: Default + Clone + Copy + Add<Output = T> + AddAssign<T> + Sub<Output = T> + SubAssign<T> + Mul<Output = T> + MulAssign<T>> EtlValueType for T {}

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
}

// It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do. Therefore, I settled on at instead, which should work fine

// TODO: See if there is any way to remove the phantom data here
pub struct EtlWrapper<T: EtlValueType, SubExpr: EtlExpr<T>> {
    pub value: SubExpr,
    pub _marker: std::marker::PhantomData<T>,
}

pub trait EtlWrappable<T: EtlValueType> {
    type WrappedAs: EtlExpr<T>;
    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs>;
}

pub trait WrappableExpr<T: EtlValueType>: EtlExpr<T> + EtlWrappable<T> {}
impl<T: EtlValueType, TT: EtlExpr<T> + EtlWrappable<T>> WrappableExpr<T> for TT {}
