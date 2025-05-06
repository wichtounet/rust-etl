use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of ReluDerivativeExpr

pub struct ReluDerivativeExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of ReluDerivativeExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> ReluDerivativeExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// ReluDerivativeExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for ReluDerivativeExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = simple_unary_type(Expr::TYPE);
    const THREAD_SAFE: bool = Expr::THREAD_SAFE;

    fn size(&self) -> usize {
        self.expr.value.size()
    }

    fn rows(&self) -> usize {
        self.expr.value.rows()
    }

    fn columns(&self) -> usize {
        self.expr.value.columns()
    }

    fn at(&self, i: usize) -> T {
        if self.expr.value.at(i) > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }

    fn at2(&self, row: usize, column: usize) -> T {
        if self.expr.value.at2(row, column) > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}

// ReluDerivativeExpr is an EtlWrappable
// ReluDerivativeExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for ReluDerivativeExpr<T, Expr> {
    type WrappedAs = ReluDerivativeExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// ReluDerivativeExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for ReluDerivativeExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

pub fn relu_derivative<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> ReluDerivativeExpr<T, Expr> {
    ReluDerivativeExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, ReluDerivativeExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, ReluDerivativeExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, ReluDerivativeExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, ReluDerivativeExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, ReluDerivativeExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::relu_derivative_expr::relu_derivative;
    use crate::vector::Vector;

    #[test]
    fn basic_relu_derivative() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = -1.0;
        a[1] = 2.0;
        a[2] = -3.0;
        a[3] = 4.0;
        a[4] = 0.0;

        let expr = relu_derivative(&a);

        assert_eq!(expr.size(), 5);
        assert_eq!(expr.at(0), 0.0);

        b |= relu_derivative(&a);

        assert_eq!(b.at(0), 0.0);
        assert_eq!(b.at(1), 1.0);
        assert_eq!(b.at(2), 0.0);
        assert_eq!(b.at(3), 1.0);
        assert_eq!(b.at(4), 0.0);
    }

    #[test]
    fn basic_relu_derivative_expr() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = -1.0;
        a[1] = 2.0;
        a[2] = -3.0;
        a[3] = 4.0;
        a[4] = 0.0;

        b |= relu_derivative(&a) + relu_derivative(&a);

        assert_eq!(b.at(0), 0.0);
        assert_eq!(b.at(1), 2.0);
        assert_eq!(b.at(2), 0.0);
        assert_eq!(b.at(3), 2.0);
        assert_eq!(b.at(4), 0.0);
    }
}
