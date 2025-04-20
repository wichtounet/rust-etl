use crate::etl_expr::*;

// The declaration of ScaleExpr

pub struct ScaleExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of ScaleExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> ScaleExpr<T, LeftExpr, RightExpr> {
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

// ScaleExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for ScaleExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS > 0 { LeftExpr::DIMENSIONS } else { RightExpr::DIMENSIONS };
    const TYPE: EtlType = EtlType::Simple;

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

    fn at(&self, i: usize) -> T {
        self.lhs.value.at(i) * self.rhs.value.at(i)
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.lhs.value.at2(row, column) * self.rhs.value.at2(row, column)
    }
}

// ScaleExpr is an EtlWrappable
// ScaleExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for ScaleExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = ScaleExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// ScaleExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for ScaleExpr<T, LeftExpr, RightExpr> {
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
macro_rules! impl_scale_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Shr<RightExpr> for &'a $type {
            type Output = $crate::scale_expr::ScaleExpr<T, &'a $type, RightExpr>;

            fn shr(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<T>> std::ops::ShrAssign<RightExpr> for $type {
            fn shr_assign(&mut self, other: RightExpr) {
                validate_assign(self, &other);
                scale_assign_direct(&mut self.data, &other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_scale_op_constant {
    ($type:ty) => {
        impl<T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Shr<RightExpr> for $type {
            type Output = $crate::scale_expr::ScaleExpr<T, $type, RightExpr>;

            fn shr(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_scale_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Shr<OuterRightExpr>
            for $type
        {
            type Output = $crate::scale_expr::ScaleExpr<T, $type, OuterRightExpr>;

            fn shr(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_scale_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Shr<OuterRightExpr> for $type {
            type Output = $crate::scale_expr::ScaleExpr<T, $type, OuterRightExpr>;

            fn shr(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_scale_op_unary_expr_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Shr<OuterRightExpr> for $type {
            type Output = $crate::scale_expr::ScaleExpr<T, $type, OuterRightExpr>;

            fn shr(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

crate::impl_add_op_binary_expr!(ScaleExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(ScaleExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(ScaleExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr!(ScaleExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(ScaleExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::constant::cst;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a >> &b;

        assert_eq!(expr.size(), 8);
        assert_eq!(expr.at(0), 2);
    }

    #[test]
    fn basic_assign_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a >> &b;

        c |= expr;

        assert_eq!(c.at(0), 2);
    }

    #[test]
    fn basic_assign_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a >> &b;

        assert_eq!(c.at(0), 2);
    }

    #[test]
    fn basic_assign_constant() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 3;
        b[0] = 7;

        c |= cst(2) >> (&a >> &b);

        assert_eq!(c.at(0), 42);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a = Matrix2d::<i64>::new(4, 2);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a >> &b;

        assert_eq!(c.at(0), 2);
    }

    #[test]
    fn basic_assign_deep_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a >> &b) >> &b;

        assert_eq!(c.at(0), 4);
    }

    #[test]
    fn basic_assign_deep_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 3;
        b[0] = 2;

        c |= (&a >> &b) + (&a >> &b);

        assert_eq!(c.at(0), 12);
    }

    #[test]
    fn basic_compound_add() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        c.fill(10);

        a[0] = 1;
        b[0] = 2;

        c >>= &a + &b;

        assert_eq!(c.at(0), 30);
    }
}
