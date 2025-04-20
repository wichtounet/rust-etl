use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of SigmoidDerivativeExpr

pub struct SigmoidDerivativeExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of SigmoidDerivativeExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> SigmoidDerivativeExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// SigmoidDerivativeExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for SigmoidDerivativeExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = EtlType::Simple;

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
        self.expr.value.at(i) * (T::one() - self.expr.value.at(i))
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.expr.value.at2(row, column) * (T::one() - self.expr.value.at2(row, column))
    }
}

// SigmoidDerivativeExpr is an EtlWrappable
// SigmoidDerivativeExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for SigmoidDerivativeExpr<T, Expr> {
    type WrappedAs = SigmoidDerivativeExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SigmoidDerivativeExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for SigmoidDerivativeExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn sigmoid_derivative<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> SigmoidDerivativeExpr<T, Expr> {
    SigmoidDerivativeExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, SigmoidDerivativeExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, SigmoidDerivativeExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, SigmoidDerivativeExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, SigmoidDerivativeExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, SigmoidDerivativeExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::sigmoid_derivative_expr::sigmoid_derivative;
    use crate::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_sigmoid() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        let expr = sigmoid_derivative(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(0), 0.0);

        b |= sigmoid_derivative(&a);

        assert_relative_eq!(b.at(0), 0.0);
        assert_relative_eq!(b.at(1), -2.0);
        assert_relative_eq!(b.at(2), -6.0);
        assert_relative_eq!(b.at(3), -12.0);
        assert_relative_eq!(b.at(4), -20.0);
    }

    #[test]
    fn basic_sigmoid_expr() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        b |= sigmoid_derivative(&a) + sigmoid_derivative(&a);

        assert_relative_eq!(b.at(0), 0.0);
        assert_relative_eq!(b.at(1), -4.0);
        assert_relative_eq!(b.at(2), -12.0);
        assert_relative_eq!(b.at(3), -24.0);
        assert_relative_eq!(b.at(4), -40.0);
    }
}
