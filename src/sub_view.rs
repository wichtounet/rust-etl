use crate::etl_expr::*;

// The declaration of SubView

#[derive(Clone)]
pub struct SubView<T: EtlValueType, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    index: usize,
}

// The functions of SubView

impl<T: EtlValueType, Expr: WrappableExpr<T>> SubView<T, Expr> {
    pub fn new(expr: Expr, index: usize) -> Self {
        Self { expr: expr.wrap(), index }
    }
}

// SubView is an EtlExpr
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlExpr<T> for SubView<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS - 1;
    const TYPE: EtlType = EtlType::Value;
    const THREAD_SAFE: bool = Expr::THREAD_SAFE;

    type Iter<'x>
        = std::iter::Cloned<std::slice::Iter<'x, T>>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        self.expr.value.get_data()[self.index * self.size()..(self.index + 1) * self.size()]
            .iter()
            .cloned()
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        self.expr.value.get_data()[self.index * self.size()..(self.index + 1) * self.size()][range]
            .iter()
            .cloned()
    }

    fn size(&self) -> usize {
        self.expr.value.size() / self.expr.value.dim(0)
    }

    fn rows(&self) -> usize {
        self.expr.value.dim(1)
    }

    fn columns(&self) -> usize {
        self.expr.value.dim(2)
    }

    fn dim(&self, i: usize) -> usize {
        self.expr.value.dim(i + 1)
    }

    fn at(&self, i: usize) -> T {
        self.expr.value.at2(self.index, i)
    }

    fn at2(&self, i: usize, j: usize) -> T {
        self.expr.value.at3(self.index, i, j)
    }

    fn get_data(&self) -> &[T] {
        &self.expr.value.get_data()[self.index * self.size()..(self.index + 1) * self.size()]
    }
}

// SubView is an EtlWrappable
// SubView wraps as value
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlWrappable<T> for SubView<T, Expr> {
    type WrappedAs = SubView<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SubView computes as copy
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlComputable<T> for SubView<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

// Note: Since Rust does not allow function return type inference, it is simpler to build an
// expression type than to return the expression itself

pub fn sub<T: EtlValueType, Expr: WrappableExpr<T>>(expr: Expr, i: usize) -> SubView<T, Expr> {
    assert!(Expr::DIMENSIONS > 1, "sub can only work on matrices");
    assert!(Expr::TYPE.direct(), "sub can only work on direct expressions");
    assert!(i < expr.dim(0));

    SubView::<T, Expr>::new(expr, i)
}

crate::impl_add_op_unary_expr!(SubView<T, Expr>);
crate::impl_sub_op_unary_expr!(SubView<T, Expr>);
crate::impl_mul_op_unary_expr!(SubView<T, Expr>);
crate::impl_div_op_unary_expr!(SubView<T, Expr>);
crate::impl_scale_op_unary_expr!(SubView<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::matrix_3d::Matrix3d;
    use crate::matrix_4d::Matrix4d;
    use crate::sub_view::sub;
    use crate::vector::Vector;

    #[test]
    fn basic_sub_2d() {
        let a = Matrix2d::<i64>::new_iota(2, 5, 1);

        let expr = sub(&a, 1);

        assert_eq!(expr.size(), 5);
        assert_eq!(expr.dim(0), 5);
        assert_eq!(expr.rows(), 5);

        let mut b = Vector::<i64>::new(5);
        b |= sub(&a, 1);

        assert_eq!(b.at(0), 6);
        assert_eq!(b.at(1), 7);
        assert_eq!(b.at(2), 8);
        assert_eq!(b.at(3), 9);
        assert_eq!(b.at(4), 10);
    }

    #[test]
    fn basic_sub_2d_iter() {
        let a = Matrix2d::<i64>::new_iota(2, 5, 1);

        let expr = sub(&a, 1);
        for (n, value) in expr.iter().enumerate() {
            assert_eq!(value, (n + 6).try_into().unwrap());
        }

        for (n, value) in expr.iter_range(1..4).enumerate() {
            assert_eq!(value, (n + 7).try_into().unwrap());
            assert!(n < 3);
        }
    }

    #[test]
    fn basic_sub_3d() {
        let a = Matrix3d::<i64>::new_iota(2, 4, 3, 1);

        let expr = sub(&a, 1);

        assert_eq!(expr.size(), 12);
        assert_eq!(expr.dim(0), 4);
        assert_eq!(expr.rows(), 4);
        assert_eq!(expr.dim(1), 3);
        assert_eq!(expr.columns(), 3);

        let mut b = Matrix2d::<i64>::new(4, 3);
        b |= sub(&a, 1);

        assert_eq!(b.at(0), 13);
        assert_eq!(b.at(1), 14);
        assert_eq!(b.at(2), 15);
        assert_eq!(b.at(3), 16);
        assert_eq!(b.at(4), 17);
    }

    #[test]
    fn basic_sub_3d_iter() {
        let a = Matrix3d::<i64>::new_iota(2, 4, 3, 1);

        let expr = sub(&a, 1);
        for (n, value) in expr.iter().enumerate() {
            assert_eq!(value, (n + 13).try_into().unwrap());
        }

        for (n, value) in expr.iter_range(1..4).enumerate() {
            assert_eq!(value, (n + 14).try_into().unwrap());
            assert!(n < 3);
        }
    }

    #[test]
    fn basic_sub_sub_3d() {
        let a = Matrix3d::<i64>::new_iota(2, 4, 3, 1);

        let expr = sub(sub(&a, 1), 1);

        assert_eq!(expr.size(), 3);
        assert_eq!(expr.dim(0), 3);
        assert_eq!(expr.rows(), 3);

        let mut b = Vector::<i64>::new(3);
        b |= sub(sub(&a, 1), 1);

        assert_eq!(b.at(0), 16);
        assert_eq!(b.at(1), 17);
        assert_eq!(b.at(2), 18);
    }

    #[test]
    fn basic_sub_4d() {
        let a = Matrix4d::<i64>::new_iota(2, 4, 3, 2, 1);

        let expr = sub(&a, 1);

        assert_eq!(expr.size(), 24);
        assert_eq!(expr.dim(0), 4);
        assert_eq!(expr.dim(1), 3);
        assert_eq!(expr.dim(2), 2);

        assert_eq!(expr.rows(), 4);
        assert_eq!(expr.columns(), 3);

        let mut b = Matrix3d::<i64>::new(4, 3, 2);
        b |= sub(&a, 1);

        assert_eq!(b.at(0), 25);
        assert_eq!(b.at(1), 26);
        assert_eq!(b.at(2), 27);
        assert_eq!(b.at(3), 28);
        assert_eq!(b.at(4), 29);
    }

    #[test]
    fn basic_sub_sub_4d() {
        let a = Matrix4d::<i64>::new_iota(2, 4, 3, 2, 1);

        let expr = sub(sub(&a, 1), 0);

        assert_eq!(expr.size(), 6);
        assert_eq!(expr.dim(0), 3);
        assert_eq!(expr.dim(1), 2);

        assert_eq!(expr.rows(), 3);
        assert_eq!(expr.columns(), 2);

        let mut b = Matrix2d::<i64>::new(3, 2);
        b |= sub(sub(&a, 1), 0);

        assert_eq!(b.at(0), 25);
        assert_eq!(b.at(1), 26);
        assert_eq!(b.at(2), 27);
        assert_eq!(b.at(3), 28);
        assert_eq!(b.at(4), 29);
        assert_eq!(b.at(5), 30);
    }

    #[test]
    fn basic_sub_sub_sub_4d() {
        let a = Matrix4d::<i64>::new_iota(2, 4, 3, 2, 1);

        let expr = sub(sub(sub(&a, 1), 0), 1);

        assert_eq!(expr.size(), 2);
        assert_eq!(expr.dim(0), 2);
        assert_eq!(expr.rows(), 2);

        let mut b = Vector::<i64>::new(2);
        b |= sub(sub(sub(&a, 1), 0), 1);

        assert_eq!(b.at(0), 27);
        assert_eq!(b.at(1), 28);
    }
}
