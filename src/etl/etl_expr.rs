//TODO.2 Add more constrants to T so that we can simplify traits bounds
pub trait EtlValueType: Default + Clone + Copy + std::ops::Add + std::ops::AddAssign {}
impl<T: Default + Clone + Copy + std::ops::Add + std::ops::AddAssign> EtlValueType for T {}

pub trait EtlExpr {
    type Type;
    fn size(&self) -> usize;
    fn at(&self, i: usize) -> Self::Type;
}

// It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do. Therefore, I settled on at instead, which should work fine

pub struct EtlWrapper<Expr: EtlExpr> {
    pub value: Expr,
}

pub trait EtlWrappable {
    type WrappedAs: EtlExpr;
    fn wrap(self) -> EtlWrapper<Self::WrappedAs>;
}

pub trait WrappableExpr: EtlExpr + EtlWrappable {}
impl<T: EtlExpr + EtlWrappable> WrappableExpr for T {}
