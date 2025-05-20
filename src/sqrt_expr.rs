use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of SqrtExprj

#[derive(Clone)]
pub struct SqrtExprj<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of SqrtExprj

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> SqrtExprj<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

pub struct SqrtExprIterator<'a, T: EtlValueType, Expr: EtlExpr<T> + 'a>
where
    T: 'a,
{
    sub_iter: Expr::Iter<'a>,
}

impl<'a, T: EtlValueType + Float, Expr: EtlExpr<T>> Iterator for SqrtExprIterator<'a, T, Expr> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.sub_iter.next().map(|sub| sub.sqrt())
    }
}

// SqrtExprj is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for SqrtExprj<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = simple_unary_type(Expr::TYPE);
    const THREAD_SAFE: bool = Expr::THREAD_SAFE;

    type Iter<'x>
        = SqrtExprIterator<'x, T, Expr::WrappedAs>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        Self::Iter {
            sub_iter: self.expr.value.iter(),
        }
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        Self::Iter {
            sub_iter: self.expr.value.iter_range(range),
        }
    }

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
        self.expr.value.at(i).sqrt()
    }
}

// SqrtExprj is an EtlWrappable
// SqrtExprj wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for SqrtExprj<T, Expr> {
    type WrappedAs = SqrtExprj<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SqrtExprj computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for SqrtExprj<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

pub fn sqrt<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> SqrtExprj<T, Expr> {
    SqrtExprj::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, SqrtExprj<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, SqrtExprj<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, SqrtExprj<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, SqrtExprj<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, SqrtExprj<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl_expr::EtlExpr;
    use crate::sqrt_expr::sqrt;
    use crate::vector::Vector;

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

        let expr = sqrt(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(0), 1.0_f64.sqrt(), epsilon = 1e-6);

        b |= sqrt(&a);

        assert_relative_eq!(b.at(0), 1.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 3.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 4.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 5.0_f64.sqrt(), epsilon = 1e-6);
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

        b |= sqrt(&a) + sqrt(&a);

        assert_relative_eq!(b.at(0), 2.0 * 1.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0 * 2.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 2.0 * 3.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 2.0 * 4.0_f64.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 2.0 * 5.0_f64.sqrt(), epsilon = 1e-6);
    }
}
