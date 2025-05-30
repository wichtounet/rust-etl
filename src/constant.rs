use crate::etl_expr::*;

// The declaration of Constant<T>

#[derive(Clone)]
pub struct Constant<T: EtlValueType> {
    value: T,
}

// The functions of Constant<T>

impl<T: EtlValueType> Constant<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

pub fn cst<T: EtlValueType>(value: T) -> Constant<T> {
    Constant::<T>::new(value)
}

pub struct ConstantIterator<T: EtlValueType> {
    value: T,
}

impl<T: EtlValueType> Iterator for ConstantIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.value)
    }
}

impl<T: EtlValueType> EtlExpr<T> for Constant<T> {
    const DIMENSIONS: usize = 0;
    const TYPE: EtlType = EtlType::Value;
    const THREAD_SAFE: bool = true;

    type Iter<'x>
        = ConstantIterator<T>
    where
        T: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        ConstantIterator { value: self.value }
    }

    fn iter_range(&self, _range: std::ops::Range<usize>) -> Self::Iter<'_> {
        ConstantIterator { value: self.value }
    }

    fn size(&self) -> usize {
        0
    }

    fn rows(&self) -> usize {
        0
    }

    fn columns(&self) -> usize {
        0
    }

    fn at(&self, _i: usize) -> T {
        self.value
    }
}

// Constant<T> computes as itself
impl<T: EtlValueType> EtlComputable<T> for Constant<T> {
    fn to_data(&self) -> Vec<T> {
        panic!("to_data should not be called on Constant");
    }
}

// Constant<T> wraps as value
impl<T: EtlValueType> EtlWrappable<T> for Constant<T> {
    type WrappedAs = Constant<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Operations

crate::impl_add_op_constant!(Constant<T>);
crate::impl_sub_op_constant!(Constant<T>);
crate::impl_scale_op_constant!(Constant<T>);
crate::impl_div_op_constant!(Constant<T>);

#[cfg(test)]
mod tests {
    use crate::matrix_2d::Matrix2d;

    use super::*;

    #[test]
    fn basic() {
        let mut b = Matrix2d::<i64>::new(2, 2);

        b |= cst(1);

        assert_eq!(b[0], 1);
        assert_eq!(b[1], 1);
        assert_eq!(b[2], 1);
        assert_eq!(b[3], 1);
    }

    #[test]
    fn basic_add() {
        let mut a = Matrix2d::<i64>::new(2, 2);
        let mut b = Matrix2d::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        b |= &a + cst(1);

        assert_eq!(b[0], 4);
        assert_eq!(b[1], 10);
        assert_eq!(b[2], 28);
        assert_eq!(b[3], 43);

        b |= cst(11) + &a;

        assert_eq!(b[0], 14);
        assert_eq!(b[1], 20);
        assert_eq!(b[2], 38);
        assert_eq!(b[3], 53);

        b |= &a - cst(1);

        assert_eq!(b[0], 2);
        assert_eq!(b[1], 8);
        assert_eq!(b[2], 26);
        assert_eq!(b[3], 41);
    }

    #[test]
    fn basic_add_mixed() {
        let mut a = Matrix2d::<i64>::new(2, 2);
        let mut b = Matrix2d::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        b |= (cst(11) + &a) + (cst(100) - &a);

        assert_eq!(b[0], 14 + 97);
        assert_eq!(b[1], 20 + 91);
        assert_eq!(b[2], 38 + 73);
        assert_eq!(b[3], 53 + 58);
    }
}
