use std::sync::Mutex;

use crate::base_traits::Float;
use crate::etl_expr::*;

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

pub trait RandFloat: Float + rand::distr::uniform::SampleUniform {}

impl RandFloat for f32 {}
impl RandFloat for f64 {}

// The declaration of InvDropoutMask<T>
// Ideally, we would use RefCell to wrap the engine, but we must use Mutex, otherwise, the type is
// not sync and this would break many operations

pub struct InvDropoutMask<T: EtlValueType + RandFloat> {
    probability: T,
    engine: Mutex<SmallRng>,
}

// The functions of InvDropoutMask<T>

impl<T: EtlValueType + RandFloat> InvDropoutMask<T> {
    pub fn new(probability: T) -> Self {
        Self {
            probability,
            engine: Mutex::<SmallRng>::new(SmallRng::from_rng(&mut rand::rng())),
        }
    }

    fn next_value(&self) -> T {
        if self.engine.lock().unwrap().random_range(T::zero()..T::one()) < self.probability {
            T::zero()
        } else {
            T::one() / (T::one() - self.probability)
        }
    }
}

pub struct InvDropoutMaskIterator<'a, T: EtlValueType + RandFloat> {
    expr: &'a InvDropoutMask<T>,
}

impl<'a, T: EtlValueType + RandFloat> Iterator for InvDropoutMaskIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.expr.next_value())
    }
}

impl<T: EtlValueType + RandFloat> EtlExpr<T> for InvDropoutMask<T> {
    const DIMENSIONS: usize = 0;
    const TYPE: EtlType = EtlType::Value;
    const THREAD_SAFE: bool = false;

    type Iter<'x>
        = InvDropoutMaskIterator<'x, T>
    where
        T: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        Self::Iter { expr: self }
    }

    fn iter_range(&self, _range: std::ops::Range<usize>) -> Self::Iter<'_> {
        Self::Iter { expr: self }
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
        self.next_value()
    }

    fn at2(&self, _row: usize, _column: usize) -> T {
        self.next_value()
    }
}

// InvDropoutMask<T> computes as itself
impl<T: EtlValueType + RandFloat> EtlComputable<T> for InvDropoutMask<T> {
    fn to_data(&self) -> Vec<T> {
        panic!("to_data should not be called on InvDropoutMask");
    }
}

// InvDropoutMask<T> wraps as value
impl<T: EtlValueType + RandFloat> EtlWrappable<T> for InvDropoutMask<T> {
    type WrappedAs = InvDropoutMask<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Operations

pub fn inv_dropout_mask<T: EtlValueType + RandFloat>(probability: T) -> InvDropoutMask<T> {
    InvDropoutMask::<T>::new(probability)
}

crate::impl_add_op_constant_trait!(RandFloat, InvDropoutMask<T>);
crate::impl_sub_op_constant_trait!(RandFloat, InvDropoutMask<T>);
crate::impl_scale_op_constant_trait!(RandFloat, InvDropoutMask<T>);
crate::impl_div_op_constant_trait!(RandFloat, InvDropoutMask<T>);

#[cfg(test)]
mod tests {
    use crate::matrix_2d::Matrix2d;

    use super::*;

    #[test]
    fn basic() {
        let mut b = Matrix2d::<f64>::new(2, 2);

        b |= inv_dropout_mask(0.5);

        assert!(b[0] == 0.0 || b[0] == 1.0 / (1.0 - 0.5));
        assert!(b[1] == 0.0 || b[1] == 1.0 / (1.0 - 0.5));
        assert!(b[2] == 0.0 || b[2] == 1.0 / (1.0 - 0.5));
        assert!(b[3] == 0.0 || b[3] == 1.0 / (1.0 - 0.5));
    }
}
