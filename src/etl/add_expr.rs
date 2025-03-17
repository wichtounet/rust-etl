use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlValueType;

use std::ops::Add;

// The declaration of AddExpr<T>

pub struct AddExpr<'a, LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    lhs: &'a LeftExpr,
    rhs: &'a RightExpr,
    _marker: std::marker::PhantomData<T>,
}

// The functions of AddExpr<T>

impl<'a, LeftExpr, RightExpr, T> AddExpr<'a, LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    pub fn new(lhs: &'a LeftExpr, rhs: &'a RightExpr) -> Self {
        if lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self { lhs: lhs, rhs: rhs, _marker: std::marker::PhantomData }
    }

    pub fn size(&self) -> usize {
        self.lhs.size()
    }

    pub fn at(&self, i: usize) -> T {
        let lhs: T = self.lhs.at(i);
        let rhs: T = self.rhs.at(i);
        lhs + rhs
    }
}

// AddExpr is an EtlExpr
impl<'a, LeftExpr, RightExpr, T> EtlExpr<T> for AddExpr<'a, LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    fn size(&self) -> usize {
        self.lhs.size()
    }

    fn at(&self, i: usize) -> T {
        let lhs: T = self.lhs.at(i);
        let rhs: T = self.rhs.at(i);
        lhs + rhs
    }
}

// Operations

impl<'a, LeftExpr, RightExpr, T, OuterRightExpr> Add<&'a OuterRightExpr> for &'a AddExpr<'a, LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T>, OuterRightExpr: EtlExpr<T> {
    type Output = AddExpr<'a, AddExpr<'a, LeftExpr, RightExpr, T>, OuterRightExpr, T>;

    fn add(self, other: &'a OuterRightExpr) -> Self::Output {
        Self::Output::new(self, other)
    }
}

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::vector::Vector;
    use crate::etl::matrix::Matrix;
    use crate::etl::etl_expr::EtlExpr;

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

        c.assign(expr);

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_2() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c.assign(&a + &b);

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a: Matrix<i64> = Matrix::<i64>::new(4, 2);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c.assign(&a + &b);

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_deep() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);
        let mut c: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c.assign(&(&a + &b) + &a);

        assert_eq!(c.at(0), 4);
    }
}
