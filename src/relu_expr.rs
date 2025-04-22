use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of ReluExpr

pub struct ReluExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of ReluExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> ReluExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// ReluExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for ReluExpr<T, Expr> {
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
        if self.expr.value.at(i) > T::zero() {
            self.expr.value.at(i)
        } else {
            T::zero()
        }
    }

    fn at2(&self, row: usize, column: usize) -> T {
        if self.expr.value.at2(row, column) > T::zero() {
            self.expr.value.at2(row, column)
        } else {
            T::zero()
        }
    }
}

// ReluExpr is an EtlWrappable
// ReluExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for ReluExpr<T, Expr> {
    type WrappedAs = ReluExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// ReluExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for ReluExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

pub fn relu<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> ReluExpr<T, Expr> {
    ReluExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, ReluExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, ReluExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, ReluExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, ReluExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, ReluExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::relu_expr::relu;
    use crate::vector::Vector;

    #[test]
    fn basic_relu() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = -1.0;
        a[1] = 2.0;
        a[2] = -3.0;
        a[3] = 4.0;
        a[4] = 0.0;

        let expr = relu(&a);

        assert_eq!(expr.size(), 5);
        assert_eq!(expr.at(0), 0.0);

        b |= relu(&a);

        assert_eq!(b.at(0), 0.0);
        assert_eq!(b.at(1), 2.0);
        assert_eq!(b.at(2), 0.0);
        assert_eq!(b.at(3), 4.0);
        assert_eq!(b.at(4), 0.0);
    }

    #[test]
    fn basic_relu_expr() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = -1.0;
        a[1] = 2.0;
        a[2] = -3.0;
        a[3] = 4.0;
        a[4] = 0.0;

        b |= relu(&a) + relu(&a);

        assert_eq!(b.at(0), 0.0);
        assert_eq!(b.at(1), 4.0);
        assert_eq!(b.at(2), 0.0);
        assert_eq!(b.at(3), 8.0);
        assert_eq!(b.at(4), 0.0);
    }
}
