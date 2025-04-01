use crate::etl::etl_expr::*;
use crate::etl::matrix_2d::Matrix2d;
use crate::etl::reductions::max;
use crate::etl::reductions::sum;
use crate::etl::vector::Vector;

use super::constant::cst;
use super::exp_expr::exp;

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
    type ComputedAsVector = Vector<T>;
    type ComputedAsMatrix = Matrix2d<T>;

    fn to_vector(&self) -> EtlWrapper<T, Self::ComputedAsVector> {
        let mut vec = Vector::<T>::new(self.rows());
        assign_direct(&mut vec.data, self);
        EtlWrapper {
            value: vec,
            _marker: std::marker::PhantomData,
        }
    }

    fn to_matrix(&self) -> EtlWrapper<T, Self::ComputedAsMatrix> {
        let mut vec = Matrix2d::<T>::new(self.rows(), self.columns());
        assign_direct(&mut vec.data, self);
        EtlWrapper {
            value: vec,
            _marker: std::marker::PhantomData,
        }
    }

    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn softmax<T: EtlValueType + Float, Expr: WrappableExpr<T> + EtlComputable<T, ComputedAsVector = Vector<T>>>(expr: Expr) -> SoftmaxExpr<T, Expr> {
    // We need a  concrete type because we have no traits implementing X - constant

    let vec: Vector<T> = expr.to_vector().value;
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

    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::softmax_expr::softmax;
    use crate::etl::vector::Vector;

    use approx::assert_relative_eq;

    // TODO Add Tests
}
