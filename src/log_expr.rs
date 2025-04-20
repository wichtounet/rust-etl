use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of LogExpr

pub struct LogExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of LogExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> LogExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// LogExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for LogExpr<T, Expr> {
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
        self.expr.value.at(i).ln()
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.expr.value.at2(row, column).ln()
    }
}

// LogExpr is an EtlWrappable
// LogExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for LogExpr<T, Expr> {
    type WrappedAs = LogExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// LogExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for LogExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn log<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> LogExpr<T, Expr> {
    LogExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, LogExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, LogExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, LogExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, LogExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, LogExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::log_expr::log;
    use crate::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_log() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        let expr = log(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(0), 1.0_f64.ln(), epsilon = 1e-6);

        b |= log(&a);

        assert_relative_eq!(b.at(0), 1.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 3.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 4.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 5.0_f64.ln(), epsilon = 1e-6);
    }

    #[test]
    fn basic_log_deep() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        b |= log(&a) + log(&a);

        assert_relative_eq!(b.at(0), 2.0 * 1.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0 * 2.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 2.0 * 3.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 2.0 * 4.0_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 2.0 * 5.0_f64.ln(), epsilon = 1e-6);
    }
}
