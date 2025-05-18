use crate::etl_expr::*;

// The declaration of MinExpr

pub struct MinExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of MinExpr

fn min_impl<T: EtlValueType>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> MinExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS > 0 && RightExpr::DIMENSIONS > 0 && lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

pub struct MinExprIterator<'a, T: EtlValueType, LeftExpr: EtlExpr<T> + 'a, RightExpr: EtlExpr<T> + 'a>
where
    T: 'a,
{
    lhs_iter: LeftExpr::Iter<'a>,
    rhs_iter: RightExpr::Iter<'a>,
}

impl<'a, T: EtlValueType, LeftExpr: EtlExpr<T>, RightExpr: EtlExpr<T>> Iterator for MinExprIterator<'a, T, LeftExpr, RightExpr> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.lhs_iter.next().zip(self.rhs_iter.next()).map(|(lhs, rhs)| min_impl(lhs, rhs))
    }
}

// MinExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for MinExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS > 0 { LeftExpr::DIMENSIONS } else { RightExpr::DIMENSIONS };
    const TYPE: EtlType = simple_binary_type(LeftExpr::TYPE, RightExpr::TYPE);
    const THREAD_SAFE: bool = LeftExpr::THREAD_SAFE && RightExpr::THREAD_SAFE;

    type Iter<'x>
        = MinExprIterator<'x, T, LeftExpr::WrappedAs, RightExpr::WrappedAs>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        Self::Iter {
            lhs_iter: self.lhs.value.iter(),
            rhs_iter: self.rhs.value.iter(),
        }
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        Self::Iter {
            lhs_iter: self.lhs.value.iter_range(range.clone()),
            rhs_iter: self.rhs.value.iter_range(range.clone()),
        }
    }

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.size()
        } else {
            self.rhs.value.size()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.rows()
        } else {
            self.rhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.columns()
        } else {
            self.rhs.value.columns()
        }
    }

    fn at(&self, i: usize) -> T {
        min_impl(self.lhs.value.at(i), self.rhs.value.at(i))
    }
}

// MinExpr is an EtlWrappable
// MinExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for MinExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = MinExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// MinExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for MinExpr<T, LeftExpr, RightExpr> {
    fn to_data(&self) -> Vec<T> {
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
    }
}

// Operations

pub fn binary_min<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>(lhs: LeftExpr, rhs: RightExpr) -> MinExpr<T, LeftExpr, RightExpr> {
    MinExpr::<T, LeftExpr, RightExpr>::new(lhs, rhs)
}

crate::impl_add_op_binary_expr!(MinExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(MinExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(MinExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr!(MinExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(MinExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::min_expr::binary_min;
    use crate::vector::Vector;

    #[test]
    fn basic_min_vec() {
        let mut a = Vector::<i64>::new(3);
        let mut b = Vector::<i64>::new(3);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        b[0] = 0;
        b[1] = 5;
        b[2] = 2;

        let expr = binary_min(&a, &b);

        assert_eq!(expr.size(), 3);
        assert_eq!(expr.at(0), 0);
    }

    #[test]
    fn basic_min_vec_assign() {
        let mut a = Vector::<i64>::new(3);
        let mut b = Vector::<i64>::new(3);
        let mut c = Vector::<i64>::new(3);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        b[0] = 0;
        b[1] = 5;
        b[2] = 2;

        c |= binary_min(&a, &b);

        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 2);
        assert_eq!(c.at(2), 2);
    }

    #[test]
    fn basic_min_mat_assign() {
        let mut a = Matrix2d::<i64>::new(2, 2);
        let mut b = Matrix2d::<i64>::new(2, 2);
        let mut c = Matrix2d::<i64>::new(2, 2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = -9;

        b[0] = 0;
        b[1] = 5;
        b[2] = 2;
        b[3] = -8;

        c |= binary_min(&a, &b);

        assert_eq!(c.at2(0, 0), 0);
        assert_eq!(c.at2(0, 1), 2);
        assert_eq!(c.at2(1, 0), 2);
        assert_eq!(c.at2(1, 1), -9);
    }
}
