use crate::base_traits::Float;
use crate::etl_expr::*;
use crate::reductions::max;
use crate::reductions::sum;
use crate::vector::Vector;

use crate::constant::cst;
use crate::exp_expr::exp;

// The declaration of StableSoftmaxExpr

pub struct StableSoftmaxExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    m: T,
    s: T,
}

// The functions of StableSoftmaxExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> StableSoftmaxExpr<T, Expr> {
    pub fn new(expr: Expr, m: T, s: T) -> Self {
        Self { expr: expr.wrap(), m, s }
    }
}

// StableSoftmaxExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for StableSoftmaxExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = simple_unary_type(Expr::TYPE);

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
        (self.expr.value.at(i) - self.m).exp() / self.s
    }
}

// StableSoftmaxExpr is an EtlWrappable
// StableSoftmaxExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for StableSoftmaxExpr<T, Expr> {
    type WrappedAs = StableSoftmaxExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// StableSoftmaxExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for StableSoftmaxExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn stable_softmax<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> StableSoftmaxExpr<T, Expr> {
    // We need a  concrete type because we have no traits implementing X - constant

    let vec = Vector::<T>::new_from_expr(&expr);
    let m = max(&vec).expect("Invalid expression for softmax");
    let s = sum(&exp(&vec - cst(m)));

    StableSoftmaxExpr::<T, Expr>::new(expr, m, s)
}

crate::impl_add_op_unary_expr_trait!(Float, StableSoftmaxExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, StableSoftmaxExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, StableSoftmaxExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, StableSoftmaxExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, StableSoftmaxExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::stable_softmax_expr::stable_softmax;
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

        b |= stable_softmax(&a);

        assert_relative_eq!(b.at(0), 0.011656, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 0.031684, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 0.086128, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 0.234121, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 0.636408, epsilon = 1e-6);
    }
}
