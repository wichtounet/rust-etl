use crate::etl::etl_expr::*;
use crate::etl::sub_expr::SubExpr;

use crate::impl_sub_op_binary_expr;

// The declaration of AddExpr

pub struct AddExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of AddExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> AddExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

// AddExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for AddExpr<T, LeftExpr, RightExpr> {
    fn size(&self) -> usize {
        self.lhs.value.size()
    }

    fn at(&self, i: usize) -> T {
        self.lhs.value.at(i) + self.rhs.value.at(i)
    }
}

// AddExpr is an EtlWrappable
// AddExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for AddExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = AddExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Operations

// Unfortunately, because of the Orphan rule, we cannot implement this trait for each structure
// implementing EtlExpr
// Therefore, we provide macros for other structures and expressions

#[macro_export]
macro_rules! impl_add_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Add<RightExpr> for &'a $type {
            type Output = AddExpr<T, &'a $type, RightExpr>;

            fn add(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<T>> std::ops::AddAssign<RightExpr> for $type {
            fn add_assign(&mut self, other: RightExpr) {
                self.add_assign_direct(other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Add<OuterRightExpr>
            for $type
        {
            type Output = AddExpr<T, $type, OuterRightExpr>;

            fn add(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

impl_add_op_binary_expr!(AddExpr<T, LeftExpr, RightExpr>);
impl_sub_op_binary_expr!(AddExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::matrix_2d::Matrix;
    use crate::etl::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a + &b;

        assert_eq!(expr.size(), 8);
        assert_eq!(expr.at(0), 3);
    }

    #[test]
    fn basic_assign_1() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a + &b;

        c |= expr;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_2() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a + &b;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a: Matrix<i64> = Matrix::<i64>::new(4, 2);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a + &b;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_deep_1() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a + &b) + &a;

        assert_eq!(c.at(0), 4);
    }

    #[test]
    fn basic_assign_deep_2() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a + &b) + (&a + &b);

        assert_eq!(c.at(0), 6);
    }

    #[test]
    fn basic_compound_add() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c += &a + &b;

        assert_eq!(c.at(0), 3);
    }
}
