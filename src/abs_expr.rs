use crate::base_traits::Abs;
use crate::etl_expr::*;

// The declaration of AbsExpr

pub struct AbsExpr<T: EtlValueType + Abs, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of AbsExpr

impl<T: EtlValueType + Abs, Expr: WrappableExpr<T>> AbsExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

pub struct AbsExprIterator<'a, T: EtlValueType, Expr: EtlExpr<T> + 'a>
where
    T: 'a,
{
    sub_iter: Expr::Iter<'a>,
}

impl<'a, T: EtlValueType + Abs, Expr: EtlExpr<T>> Iterator for AbsExprIterator<'a, T, Expr> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.sub_iter.next().map(|sub| sub.abs())
    }
}

// AbsExpr is an EtlExpr
impl<T: EtlValueType + Abs, Expr: WrappableExpr<T>> EtlExpr<T> for AbsExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = simple_unary_type(Expr::TYPE);
    const THREAD_SAFE: bool = Expr::THREAD_SAFE;

    type Iter<'x>
        = AbsExprIterator<'x, T, Expr::WrappedAs>
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
        self.expr.value.at(i).abs()
    }
}

// AbsExpr is an EtlWrappable
// AbsExpr wraps as value
impl<T: EtlValueType + Abs, Expr: WrappableExpr<T>> EtlWrappable<T> for AbsExpr<T, Expr> {
    type WrappedAs = AbsExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// AbsExpr computes as copy
impl<T: EtlValueType + Abs, Expr: WrappableExpr<T>> EtlComputable<T> for AbsExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn abs<T: EtlValueType + Abs, Expr: WrappableExpr<T>>(expr: Expr) -> AbsExpr<T, Expr> {
    AbsExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Abs, AbsExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Abs, AbsExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Abs, AbsExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Abs, AbsExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Abs, AbsExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::abs_expr::abs;
    use crate::etl_expr::EtlExpr;
    use crate::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_exp() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = -2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        let expr = abs(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(1), 2.0, epsilon = 1e-6);

        b |= abs(&a);

        assert_relative_eq!(b.at(0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 3.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 4.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 5.0, epsilon = 1e-6);
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

        b |= abs(&a) + abs(&a);

        assert_relative_eq!(b.at(0), 2.0 * 1.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 2.0 * 2.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 2.0 * 3.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 2.0 * 4.0, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 2.0 * 5.0, epsilon = 1e-6);
    }
}
