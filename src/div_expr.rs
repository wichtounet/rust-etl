use crate::etl_expr::*;

// The declaration of DivExpr

#[derive(Clone)]
pub struct DivExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of DivExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> DivExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS > 0 && RightExpr::DIMENSIONS > 0 && lhs.size() != rhs.size() {
            panic!("Cannot divide expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

pub struct DivExprIterator<'a, T: EtlValueType, LeftExpr: EtlExpr<T> + 'a, RightExpr: EtlExpr<T> + 'a>
where
    T: 'a,
{
    lhs_iter: LeftExpr::Iter<'a>,
    rhs_iter: RightExpr::Iter<'a>,
}

impl<'a, T: EtlValueType, LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>> Iterator for DivExprIterator<'a, T, LeftExpr, RightExpr> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.lhs_iter.next().zip(self.rhs_iter.next()).map(|(lhs, rhs)| lhs / rhs)
    }
}

// DivExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for DivExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS > 0 { LeftExpr::DIMENSIONS } else { RightExpr::DIMENSIONS };
    const TYPE: EtlType = EtlType::Unaligned; // To avoid divisions by zero
    const THREAD_SAFE: bool = LeftExpr::THREAD_SAFE && RightExpr::THREAD_SAFE;

    type Iter<'x>
        = DivExprIterator<'x, T, LeftExpr::WrappedAs, RightExpr::WrappedAs>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        Self::Iter {
            lhs_iter: self.lhs.value.iter(),
            rhs_iter: self.rhs.value.iter(),
        }
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        Self::Iter {
            lhs_iter: self.lhs.value.iter_range(range.clone()),
            rhs_iter: self.rhs.value.iter_range(range.clone()),
        }
    }

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.size()
        } else {
            self.rhs.value.size()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.rows()
        } else {
            self.rhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.columns()
        } else {
            self.rhs.value.columns()
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.lhs.value.at(i) / self.rhs.value.at(i)
    }
}

// DivExpr is an EtlWrappable
// DivExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for DivExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = DivExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// DivExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for DivExpr<T, LeftExpr, RightExpr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Unfortunately, because of the Orphan rule, we cannot implement this trait for each structure
// implementing EtlExpr
// Therefore, we provide macros for other structures and expressions

#[macro_export]
macro_rules! impl_div_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Div<RightExpr> for &'a $type {
            type Output = $crate::div_expr::DivExpr<T, &'a $type, RightExpr>;

            fn div(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<T>> std::ops::DivAssign<RightExpr> for $type {
            fn div_assign(&mut self, other: RightExpr) {
                validate_assign(self, &other);
                div_assign_direct(&mut self.data, &other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_constant {
    ($type:ty) => {
        impl<T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Div<RightExpr> for $type {
            type Output = $crate::div_expr::DivExpr<T, $type, RightExpr>;

            fn div(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_constant_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, RightExpr: WrappableExpr<T>> std::ops::Div<RightExpr> for $type {
            type Output = $crate::div_expr::DivExpr<T, $type, RightExpr>;

            fn div(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Div<OuterRightExpr>
            for $type
        {
            type Output = $crate::div_expr::DivExpr<T, $type, OuterRightExpr>;

            fn div(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_binary_expr_simd {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Div<OuterRightExpr> for $type
        where
            Simd<T, 8>: SimdHelper,
        {
            type Output = $crate::div_expr::DivExpr<T, $type, OuterRightExpr>;

            fn div(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Div<OuterRightExpr> for $type {
            type Output = $crate::div_expr::DivExpr<T, $type, OuterRightExpr>;

            fn div(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_div_op_unary_expr_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Div<OuterRightExpr> for $type {
            type Output = $crate::div_expr::DivExpr<T, $type, OuterRightExpr>;

            fn div(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

crate::impl_add_op_binary_expr!(DivExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr!(DivExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(DivExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(DivExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(DivExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        let expr = &a / &b;

        assert_eq!(expr.size(), 2);
        assert_eq!(expr.at(0), 0);
    }

    #[test]
    fn basic_assign_1() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        let expr = &a / &b;

        c |= expr;

        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 1);
    }

    #[test]
    fn basic_assign_2() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        c |= &a / &b;

        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 1);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a = Matrix2d::<i64>::new(2, 1);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        c |= &a / &b;

        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 1);
    }

    #[test]
    fn basic_assign_deep_1() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        c |= (&a / &b) / &a;

        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 0);
    }

    #[test]
    fn basic_assign_deep_2() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 3;

        b[0] = 2;
        b[1] = 2;

        c |= (&a + &b) / (&a + &b);

        assert_eq!(c.at(0), 1);
        assert_eq!(c.at(1), 1);
    }

    #[test]
    fn basic_compound_div() {
        let mut a = Vector::<i64>::new(2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 12;
        a[1] = 6;

        b[0] = 2;
        b[1] = 2;

        c[0] = 21;
        c[1] = 29;

        c /= &a / &b;

        assert_eq!(c.at(0), 3);
        assert_eq!(c.at(1), 9);
    }
}
