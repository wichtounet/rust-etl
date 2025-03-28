use crate::etl::etl_expr::*;
use crate::etl::matrix_2d::Matrix2d;
use crate::etl::vector::Vector;

// The declaration of ExprExpr

pub struct ExprExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of ExprExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> ExprExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// ExprExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for ExprExpr<T, Expr> {
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
        self.expr.value.at(i).exp()
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.expr.value.at2(row, column).exp()
    }
}

// ExprExpr is an EtlWrappable
// ExprExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for ExprExpr<T, Expr> {
    type WrappedAs = ExprExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// ExprExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for ExprExpr<T, Expr> {
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
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn exp<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> ExprExpr<T, Expr> {
    ExprExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_float!(ExprExpr<T, Expr>);
crate::impl_sub_op_unary_expr_float!(ExprExpr<T, Expr>);
crate::impl_mul_op_unary_expr_float!(ExprExpr<T, Expr>);
crate::impl_scale_op_unary_expr_float!(ExprExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::exp_expr::exp;
    use crate::etl::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_exp() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        let expr = exp(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(0), 1.0_f64.exp(), epsilon = 1e-6);

        b |= exp(&a);

        assert_relative_eq!(b.at(0), 1.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 3.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 4.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 5.0_f64.exp(), epsilon = 1e-6);
    }

    #[test]
    fn basic_exp_deep() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        b |= exp(&a) + exp(&a);

        assert_relative_eq!(b.at(0), 2.0 * 1.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0 * 2.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 2.0 * 3.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 2.0 * 4.0_f64.exp(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 2.0 * 5.0_f64.exp(), epsilon = 1e-6);
    }
}
