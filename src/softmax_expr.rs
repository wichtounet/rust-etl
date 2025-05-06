use crate::base_traits::Float;
use crate::etl_expr::*;
use crate::reductions::sum;
use crate::vector::Vector;

use crate::exp_expr::exp;

// The declaration of SoftmaxExpr

pub struct SoftmaxExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    s: T,
}

// The functions of SoftmaxExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> SoftmaxExpr<T, Expr> {
    pub fn new(expr: Expr, s: T) -> Self {
        Self { expr: expr.wrap(), s }
    }
}

// SoftmaxExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for SoftmaxExpr<T, Expr> {
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
        self.expr.value.at(i).exp() / self.s
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
    let s = sum(&exp(&vec));

    SoftmaxExpr::<T, Expr>::new(expr, s)
}

crate::impl_add_op_unary_expr_trait!(Float, SoftmaxExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, SoftmaxExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, SoftmaxExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, SoftmaxExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, SoftmaxExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::softmax_expr::softmax;
    use crate::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_softmax() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        b |= softmax(&a);

        assert_relative_eq!(b.at(0), 0.011656, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 0.031684, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 0.086128, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 0.234121, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 0.636408, epsilon = 1e-6);
    }
}
