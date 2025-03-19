use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlWrappable;
use crate::etl::etl_expr::EtlWrapper;
use crate::etl::etl_expr::WrappableExpr;

use std::ops::Add;

// The declaration of AddExpr

pub struct AddExpr<LeftExpr: WrappableExpr, RightExpr: WrappableExpr> {
    lhs: EtlWrapper<LeftExpr::WrappedAs>,
    rhs: EtlWrapper<RightExpr::WrappedAs>,
}

// The functions of AddExpr

impl<LeftExpr: WrappableExpr, RightExpr: WrappableExpr> AddExpr<LeftExpr, RightExpr> {
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
impl<LeftExpr: WrappableExpr, RightExpr: WrappableExpr> EtlExpr for AddExpr<LeftExpr, RightExpr>
where
    <LeftExpr::WrappedAs as EtlExpr>::Type: Add<<RightExpr::WrappedAs as EtlExpr>::Type, Output = <LeftExpr::WrappedAs as EtlExpr>::Type>,
{
    type Type = <LeftExpr::WrappedAs as EtlExpr>::Type;

    fn size(&self) -> usize {
        self.lhs.value.size()
    }

    fn at(&self, i: usize) -> Self::Type {
        self.lhs.value.at(i) + self.rhs.value.at(i)
    }
}

// AddExpr is an EtlWrappable
// AddExpr wraps as value
impl<LeftExpr: WrappableExpr, RightExpr: WrappableExpr> EtlWrappable for AddExpr<LeftExpr, RightExpr>
where
    <LeftExpr::WrappedAs as EtlExpr>::Type: Add<<RightExpr::WrappedAs as EtlExpr>::Type, Output = <LeftExpr::WrappedAs as EtlExpr>::Type>,
{
    type WrappedAs = AddExpr<LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<Self::WrappedAs> {
        EtlWrapper { value: self }
    }
}

// Operations

// Unfortunately, because of the Orphan rule, we cannot implement this trait for each structure
// implementing EtlExpr
// Therefore, we provide macros for other structures and expressions

#[macro_export]
macro_rules! impl_add_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr> std::ops::Add<RightExpr> for &'a $type
        where
            T: Add<RightExpr::Type, Output = T>,
        {
            type Output = AddExpr<&'a $type, RightExpr>;

            fn add(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<Type = T>> std::ops::AddAssign<RightExpr> for $type
        where
            T: AddAssign<RightExpr::Type>,
        {
            fn add_assign(&mut self, other: RightExpr) {
                self.add_assign_direct(other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_binary_expr {
    ($type:ty) => {
        impl<LeftExpr: WrappableExpr, RightExpr: WrappableExpr, OuterRightExpr: WrappableExpr> Add<OuterRightExpr> for $type
        where
            <LeftExpr::WrappedAs as EtlExpr>::Type: Add<<RightExpr::WrappedAs as EtlExpr>::Type, Output = <LeftExpr::WrappedAs as EtlExpr>::Type>,
        {
            type Output = AddExpr<$type, OuterRightExpr>;

            fn add(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

impl_add_op_binary_expr!(AddExpr<LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::matrix::Matrix;
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
