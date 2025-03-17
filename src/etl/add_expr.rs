use crate::etl::etl_expr::EtlExpr;

// The declaration of AddExpr<T>

pub struct AddExpr<LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: std::ops::Add<Output = T> {
    lhs: LeftExpr,
    rhs: RightExpr,
    _marker: std::marker::PhantomData<T>,
}

// The functions of AddExpr<T>

impl<LeftExpr, RightExpr, T> AddExpr<LeftExpr, RightExpr, T> where LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>, T: std::ops::Add<Output = T> {
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

#[cfg(test)]
mod tests {
    use crate::etl::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a: Vector<i64> = Vector::<i64>::new(8);
        let mut b: Vector<i64> = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = a + b;

        assert_eq!(expr.size(), 8);
        assert_eq!(expr.at(0), 3);

        // TODO
    }

}
