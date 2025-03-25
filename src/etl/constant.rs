use crate::etl::add_expr::AddExpr;
use crate::etl::etl_expr::*;
use crate::etl::mul_expr::MulExpr;
use crate::etl::sub_expr::SubExpr;

use crate::impl_add_op_value;
use crate::impl_mul_op_value;
use crate::impl_sub_op_value;

use crate::etl::vector::Vector;

use super::matrix_2d::Matrix2d;

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

// Matrix2d<T> computes as itself
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

//impl_add_op_value!(Constant<T>);
//impl_sub_op_value!(Constant<T>);
//impl_mul_op_value!(Constant<T>);

#[cfg(test)]
mod tests {
    use crate::etl::matrix_2d::Matrix2d;

    use super::*;

    #[test]
    fn basic() {
        let mut a = Matrix2d::<i64>::new(2, 2);
        let mut b = Matrix2d::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        b |= cst(1);

        assert_eq!(b[0], 1);
        assert_eq!(b[1], 1);
        assert_eq!(b[2], 1);
        assert_eq!(b[3], 1);
    }
}
