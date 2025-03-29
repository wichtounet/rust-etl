use super::etl_expr::*;
use super::matrix_2d::Matrix2d;
use super::vector::Vector;

// The declaration of Constant<T>

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

impl<T: EtlValueType> EtlExpr<T> for Constant<T> {
    const DIMENSIONS: usize = 0;
    const TYPE: EtlType = EtlType::Value;

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

    fn at2(&self, _row: usize, _column: usize) -> T {
        self.value
    }
}

// Constant<T> computes as itself
impl<T: EtlValueType> EtlComputable<T> for Constant<T> {
    type ComputedAsVector = Vector<T>;
    type ComputedAsMatrix = Matrix2d<T>;

    fn to_vector(&self) -> EtlWrapper<T, Self::ComputedAsVector> {
        panic!("to_vector should not be called on Constant");
    }

    fn to_matrix(&self) -> EtlWrapper<T, Self::ComputedAsMatrix> {
        panic!("to_matrix should not be called on Constant");
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

#[cfg(test)]
mod tests {
    use crate::etl::matrix_2d::Matrix2d;

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
