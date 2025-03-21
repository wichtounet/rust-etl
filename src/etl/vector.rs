use crate::etl::add_expr::AddExpr;
use crate::etl::etl_expr::*;
use crate::etl::mul_expr::MulExpr;
use crate::etl::sub_expr::SubExpr;

use crate::impl_add_op_value;
use crate::impl_mul_op_value;
use crate::impl_sub_op_value;

use std::ops::BitOrAssign;

use rand::Rng;

// The declaration of Vector<T>

pub struct Vector<T: EtlValueType> {
    data: Vec<T>,
}

// The functions of Vector<T>

impl<T: EtlValueType> Vector<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![T::default(); size],
        }
    }

    pub fn new_rand(size: usize) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut vec = Self::new(size);
        vec.rand_fill();
        vec
    }

    pub fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    pub fn assign_direct<RightExpr: EtlExpr<T>>(&mut self, rhs: RightExpr) {
        for i in 0..self.size() {
            self.data[i] = rhs.at(i);
        }
    }

    pub fn add_assign_direct<RightExpr: EtlExpr<T>>(&mut self, rhs: RightExpr) {
        for i in 0..self.size() {
            self.data[i] += rhs.at(i);
        }
    }

    pub fn sub_assign_direct<RightExpr: EtlExpr<T>>(&mut self, rhs: RightExpr) {
        for i in 0..self.size() {
            self.data[i] -= rhs.at(i);
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.size() {
            self.data[i] = T::default();
        }
    }

    pub fn fill(&mut self, value: T) {
        for i in 0..self.size() {
            self.data[i] = value;
        }
    }

    pub fn rand_fill(&mut self)
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut rng = rand::rng();

        for i in 0..self.size() {
            self.data[i] = rng.random::<T>();
        }
    }

    pub fn iter(&self) -> VectorIterator<T> {
        VectorIterator::<T> { vector: self, index: 0 }
    }

    // Writing my own mutable iterator requires unsafe code (which I should do later)
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: EtlValueType> EtlExpr<T> for Vector<T> {
    const DIMENSIONS: usize = 1;

    fn size(&self) -> usize {
        self.data.len()
    }

    fn rows(&self) -> usize {
        self.data.len()
    }

    fn at(&self, i: usize) -> T {
        self.data[i]
    }
}

impl<T: EtlValueType> EtlExpr<T> for &Vector<T> {
    const DIMENSIONS: usize = 1;

    fn size(&self) -> usize {
        self.data.len()
    }

    fn rows(&self) -> usize {
        self.data.len()
    }

    fn at(&self, i: usize) -> T {
        self.data[i]
    }
}

// Vector<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable<T> for &'a Vector<T> {
    type WrappedAs = &'a Vector<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: &self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Operator overloading for Vector<T>

impl<T: EtlValueType> std::ops::Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: EtlValueType> std::ops::IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// Since we can't overload Assign, we settle for BitOrAssign
impl<T: EtlValueType, RightExpr: EtlExpr<T>> BitOrAssign<RightExpr> for Vector<T> {
    fn bitor_assign(&mut self, other: RightExpr) {
        self.assign_direct(other);
    }
}

// The declaration of VectorIterator<T>

pub struct VectorIterator<'a, T: EtlValueType> {
    vector: &'a Vector<T>,
    index: usize,
}

// The implementation of VectorIterator<T>

impl<'a, T: EtlValueType> Iterator for VectorIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vector.size() {
            let result = Some(&self.vector[self.index]);
            self.index += 1;
            result
        } else {
            None
        }
    }
}

// Operations

impl_add_op_value!(Vector<T>);
impl_sub_op_value!(Vector<T>);
impl_mul_op_value!(Vector<T>);

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let vec = Vector::<i64>::new(8);
        assert_eq!(vec.size(), 8);
        assert_eq!(vec.rows(), 8);
    }

    #[test]
    fn construct_f64() {
        let vec = Vector::<f64>::new(1023);
        assert_eq!(vec.size(), 1023);
    }

    #[test]
    fn default_value() {
        let mat = Vector::<f64>::new(8);

        assert_eq!(mat.at(0), 0.0);
        assert_eq!(mat.at(1), 0.0);
        assert_eq!(mat.at(2), 0.0);
    }

    #[test]
    fn at() {
        let mut vec = Vector::<i64>::new(3);

        vec[0] = 9;
        vec[1] = 3;
        vec[2] = 7;

        assert_eq!(vec.at(0), 9);
        assert_eq!(vec.at(1), 3);
        assert_eq!(vec.at(2), 7);

        *vec.at_mut(1) = 77;
        assert_eq!(vec.at(1), 77);
    }

    #[test]
    fn fill() {
        let mut mat = Vector::<i64>::new(3);
        mat.fill(9);

        assert_eq!(mat.at(0), 9);
        assert_eq!(mat.at(1), 9);
        assert_eq!(mat.at(2), 9);
    }

    #[test]
    fn clear() {
        let mut mat = Vector::<i64>::new(3);
        mat.fill(9);

        assert_eq!(mat.at(0), 9);
        assert_eq!(mat.at(1), 9);
        assert_eq!(mat.at(2), 9);

        mat.clear();

        assert_eq!(mat.at(0), 0);
        assert_eq!(mat.at(1), 0);
        assert_eq!(mat.at(2), 0);
    }

    #[test]
    fn compound() {
        let mut a = Vector::<i64>::new(3);
        let mut b = Vector::<i64>::new(3);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;

        b[0] = 2;
        b[1] = 4;
        b[2] = 16;

        a += b;

        assert_eq!(a.at(0), 5);
        assert_eq!(a.at(1), 13);
        assert_eq!(a.at(2), 43);

        assert_eq!(a[0], 5);
        assert_eq!(a[1], 13);
        assert_eq!(a[2], 43);
    }
}
