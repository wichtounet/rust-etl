use crate::etl::etl_expr::EtlExpr;

use std::ops::Add;

// The declaration of AddExpr

pub struct AddExpr<'a, LeftExpr, RightExpr>
where LeftExpr: EtlExpr, RightExpr: EtlExpr {
    lhs: &'a LeftExpr,
    rhs: &'a RightExpr
}

// The functions of AddExpr

impl<'a, LeftExpr, RightExpr> AddExpr<'a, LeftExpr, RightExpr>
where LeftExpr: EtlExpr, RightExpr: EtlExpr {
    pub fn new(lhs: &'a LeftExpr, rhs: &'a RightExpr) -> Self {
        if lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self { lhs: lhs, rhs: rhs }
    }
}

// AddExpr is an EtlExpr
impl<'a, LeftExpr, RightExpr> EtlExpr for AddExpr<'a, LeftExpr, RightExpr>
where LeftExpr: EtlExpr, RightExpr: EtlExpr, LeftExpr::Type: Add<RightExpr::Type, Output = LeftExpr::Type> {
    type Type = LeftExpr::Type;

    fn size(&self) -> usize {
        self.lhs.size()
    }

    fn at(&self, i: usize) -> Self::Type {
        let lhs: LeftExpr::Type = self.lhs.at(i);
        let rhs: RightExpr::Type = self.rhs.at(i);
        lhs + rhs
    }
}

// Operations

impl<'a, LeftExpr, RightExpr, OuterRightExpr> Add<&'a OuterRightExpr> for &'a AddExpr<'a, LeftExpr, RightExpr> 
where LeftExpr: EtlExpr, RightExpr: EtlExpr, OuterRightExpr: EtlExpr, LeftExpr::Type: Add<RightExpr::Type, Output = LeftExpr::Type> {
    type Output = AddExpr<'a, AddExpr<'a, LeftExpr, RightExpr>, OuterRightExpr>;

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

        // TODO.2 The sub expression MUST be moved into AddExpr
        c.assign(&(&a + &b) + &a);

        assert_eq!(c.at(0), 4);
    }
}
