use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

pub trait EtlValueType: Default + Clone + Copy + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign {}
impl<T: Default + Clone + Copy + Add<Output = T> + AddAssign<T> + Sub<Output = T> + SubAssign<T>> EtlValueType for T {}

pub trait EtlExpr<T: EtlValueType> {
    fn size(&self) -> usize;
    fn at(&self, i: usize) -> T;
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
