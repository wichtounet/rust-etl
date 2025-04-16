use crate::base_traits::Float;
use crate::etl_expr::*;
use crate::reductions::max;
use crate::reductions::sum;
use crate::vector::Vector;

use crate::constant::cst;
use crate::exp_expr::exp;

// The declaration of SoftmaxExpr

pub struct SoftmaxExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    m: T,
    s: T,
}

// The functions of SoftmaxExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> SoftmaxExpr<T, Expr> {
    pub fn new(expr: Expr, m: T, s: T) -> Self {
        Self { expr: expr.wrap(), m, s }
    }
}

// SoftmaxExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for SoftmaxExpr<T, Expr> {
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
        (self.expr.value.at(i) - self.m) / self.s
    }
}

// SoftmaxExpr is an EtlWrappable
// SoftmaxExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for SoftmaxExpr<T, Expr> {
    type WrappedAs = SoftmaxExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SoftmaxExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for SoftmaxExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn softmax<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> SoftmaxExpr<T, Expr> {
    // We need a  concrete type because we have no traits implementing X - constant

    let vec = Vector::<T>::new_from_expr(&expr);
    let m = max(&vec).expect("Invalid expression for softmax");
    let s = sum(&exp(&vec - cst(m)));

    SoftmaxExpr::<T, Expr>::new(expr, m, s)
}

crate::impl_add_op_unary_expr_float!(SoftmaxExpr<T, Expr>);
crate::impl_sub_op_unary_expr_float!(SoftmaxExpr<T, Expr>);
crate::impl_mul_op_unary_expr_float!(SoftmaxExpr<T, Expr>);
crate::impl_scale_op_unary_expr_float!(SoftmaxExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::softmax_expr::softmax;
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

        b |= softmax(&a);

        assert_relative_eq!(b.at(0), -2.545634, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), -1.909225, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), -1.272817, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), -0.636408, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 0.0, epsilon = 1e-6);
    }
}
