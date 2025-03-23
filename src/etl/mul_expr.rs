use crate::etl::add_expr::AddExpr;
use crate::etl::etl_expr::*;
use crate::etl::sub_expr::SubExpr;

use crate::impl_add_op_binary_expr;
use crate::impl_sub_op_binary_expr;

// The declaration of MulExpr

/// Expression represneting a vector-matrix-multiplication
/// LeftExpr is a vector expression
/// RightExpr is a matrix expression
/// MulExpr is a vector expression
pub struct MulExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of MulExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> MulExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != rhs.rows() {
                panic!(
                    "Invalid vector matrix multiplication dimensions ([{}]*[{},{}])",
                    lhs.rows(),
                    rhs.rows(),
                    rhs.columns()
                );
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            if lhs.columns() != rhs.rows() {
                panic!(
                    "Invalid matrix vector multiplication dimensions ([{},{}]*[{}])",
                    lhs.rows(),
                    rhs.rows(),
                    rhs.columns()
                );
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            if lhs.columns() != rhs.rows() {
                panic!(
                    "Invalid matrix matrix multiplication dimensions ([{},{}]*[{},{}])",
                    lhs.rows(),
                    lhs.columns(),
                    rhs.rows(),
                    rhs.columns()
                );
            }
        } else {
            panic!(
                "Invalid vector matrix multiplication dimensions ({}D*{}D)",
                LeftExpr::DIMENSIONS,
                RightExpr::DIMENSIONS
            );
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

// MulExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for MulExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 { 2 } else { 1 };

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            self.rhs.value.columns()
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            self.lhs.value.rows()
        } else {
            self.lhs.value.rows() * self.rhs.value.columns()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            self.rhs.value.columns()
        } else {
            self.lhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        self.rhs.value.columns()
    }

    fn at(&self, i: usize) -> T {
        // TODO: Do a lazy computation

        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            let mut value = T::default();

            for r in 0..self.rhs.value.rows() {
                value += self.lhs.value.at(r) * self.rhs.value.at2(r, i)
            }

            value
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            let mut value = T::default();

            for c in 0..self.lhs.value.columns() {
                value += self.lhs.value.at2(i, c) * self.rhs.value.at(c);
            }

            value
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            self.at2(i / self.columns(), i % self.columns())
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn at2(&self, row: usize, column: usize) -> T {
        // TODO: Do a lazy computation

        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            let mut value = T::default();

            for inner in 0..self.lhs.value.columns() {
                value += self.lhs.value.at2(row, inner) * self.rhs.value.at2(inner, column)
            }

            value
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// MulExpr is an EtlWrappable
// TODO MulExpr wraps as reference?
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for MulExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = MulExpr<T, LeftExpr, RightExpr>;

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

// Unfortunately, "associated const equality" is an incomplete feature in Rust (issue 92827)
// Therefore, we need to use the same struct for each multiplication and then use if statements to
// detect the actual operation (gemm, gemv, gemv)

#[macro_export]
macro_rules! impl_mul_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Mul<RightExpr> for &'a $type {
            type Output = MulExpr<T, &'a $type, RightExpr>;

            fn mul(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr>
            for $type
        {
            type Output = MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

impl_add_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);
impl_sub_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);
impl_mul_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::matrix_2d::Matrix2d;
    use crate::etl::vector::Vector;

    // TODO These tests are really bad

    #[test]
    fn basic_gevm_one() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        c |= &a * &b;

        assert_eq!(c.at(0), 24);
    }

    #[test]
    fn basic_gevm_one_plus() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at(0), 48);
    }

    #[test]
    fn basic_gevm_one_plus_plus() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at(0), 96);
    }

    #[test]
    fn basic_gemv_one() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        c |= &a * &b;

        assert_eq!(c.at(0), 48);
    }

    #[test]
    fn basic_gemv_one_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at(0), 96);
    }

    #[test]
    fn basic_gemv_one_plus_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at(0), 192);
    }

    #[test]
    fn basic_gemm_one() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        c |= &a * &b;

        assert_eq!(c.at2(0, 0), 48);
    }

    #[test]
    fn basic_gemm_one_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at2(0, 0), 96);
    }

    #[test]
    fn basic_gemm_one_plus_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at2(0, 0), 192);
    }
}
