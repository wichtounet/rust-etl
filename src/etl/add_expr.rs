use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlValueType;

use std::ops::Add;

// The declaration of AddExpr<T>

pub struct AddExpr<LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    lhs: LeftExpr,
    rhs: RightExpr,
    _marker: std::marker::PhantomData<T>,
}

// The functions of AddExpr<T>

impl<LeftExpr, RightExpr, T> AddExpr<LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
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

impl<LeftExpr, RightExpr, T> EtlExpr<T> for AddExpr<LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    fn size(&self) -> usize {
        self.lhs.size()
    }

    fn at(&self, i: usize) -> T {
        let lhs: T = self.lhs.at(i);
        let rhs: T = self.rhs.at(i);
        lhs + rhs
    }
}

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::vector::Vector;
    use crate::etl::etl_expr::EtlExpr;

    #[test]
    fn basic_one() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = a + b;

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

        let expr = a + b;

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

        c.assign(a + b);

        assert_eq!(c.at(0), 3);
    }
}
