use crate::etl_expr::*;

// The declaration of SubExpr

#[derive(Clone)]
pub struct SubExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of SubExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> SubExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS > 0 && RightExpr::DIMENSIONS > 0 && lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

pub struct SubExprIterator<'a, T: EtlValueType, LeftExpr: EtlExpr<T> + 'a, RightExpr: EtlExpr<T> + 'a>
where
    T: 'a,
{
    lhs_iter: LeftExpr::Iter<'a>,
    rhs_iter: RightExpr::Iter<'a>,
}

impl<'a, T: EtlValueType, LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>> Iterator for SubExprIterator<'a, T, LeftExpr, RightExpr> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.lhs_iter.next().zip(self.rhs_iter.next()).map(|(lhs, rhs)| lhs - rhs)
    }
}

// SubExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for SubExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS > 0 { LeftExpr::DIMENSIONS } else { RightExpr::DIMENSIONS };
    const TYPE: EtlType = simple_binary_type(LeftExpr::TYPE, RightExpr::TYPE);
    const THREAD_SAFE: bool = LeftExpr::THREAD_SAFE && RightExpr::THREAD_SAFE;

    type Iter<'x>
        = SubExprIterator<'x, T, LeftExpr::WrappedAs, RightExpr::WrappedAs>
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
        self.lhs.value.at(i) - self.rhs.value.at(i)
    }
}

// SubExpr is an EtlWrappable
// SubExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for SubExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = SubExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SubExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for SubExpr<T, LeftExpr, RightExpr> {
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
macro_rules! impl_sub_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Sub<RightExpr> for &'a $type {
            type Output = $crate::sub_expr::SubExpr<T, &'a $type, RightExpr>;

            fn sub(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<T>> std::ops::SubAssign<RightExpr> for $type {
            fn sub_assign(&mut self, other: RightExpr) {
                validate_assign(self, &other);
                sub_assign_direct(&mut self.data, &other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_constant {
    ($type:ty) => {
        impl<T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Sub<RightExpr> for $type {
            type Output = $crate::sub_expr::SubExpr<T, $type, RightExpr>;

            fn sub(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_constant_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, RightExpr: WrappableExpr<T>> std::ops::Sub<RightExpr> for $type {
            type Output = $crate::sub_expr::SubExpr<T, $type, RightExpr>;

            fn sub(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Sub<OuterRightExpr>
            for $type
        {
            type Output = $crate::sub_expr::SubExpr<T, $type, OuterRightExpr>;

            fn sub(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_binary_expr_simd {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Sub<OuterRightExpr> for $type
        where
            Simd<T, 8>: SimdHelper,
        {
            type Output = $crate::sub_expr::SubExpr<T, $type, OuterRightExpr>;

            fn sub(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Sub<OuterRightExpr> for $type {
            type Output = $crate::sub_expr::SubExpr<T, $type, OuterRightExpr>;

            fn sub(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_sub_op_unary_expr_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Sub<OuterRightExpr> for $type {
            type Output = $crate::sub_expr::SubExpr<T, $type, OuterRightExpr>;

            fn sub(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

crate::impl_add_op_binary_expr!(SubExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(SubExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(SubExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr!(SubExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(SubExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a - &b;

        assert_eq!(expr.size(), 8);
        assert_eq!(expr.at(0), -1);
    }

    #[test]
    fn basic_assign_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a - &b;

        c |= expr;

        assert_eq!(c.at(0), -1);
    }

    #[test]
    fn basic_assign_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a - &b;

        assert_eq!(c.at(0), -1);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a = Matrix2d::<i64>::new(4, 2);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a - &b;

        assert_eq!(c.at(0), -1);
    }

    #[test]
    fn basic_assign_deep_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a - &b) - &a;

        assert_eq!(c.at(0), -2);
    }

    #[test]
    fn basic_assign_deep_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a + &b) - (&a - &b);

        assert_eq!(c.at(0), 4);
    }

    #[test]
    fn basic_compound_add() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c -= &a - &b;

        assert_eq!(c.at(0), 1);
    }
}
