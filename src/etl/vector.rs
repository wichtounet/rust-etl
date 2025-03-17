use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlValueType;
use crate::etl::add_expr::AddExpr;

use std::ops::Add;

// The declaration of Vector<T>

pub struct Vector<T: EtlValueType> {
    data: Vec<T>
}

// The functions of Vector<T>

impl<T: EtlValueType> Vector<T> {
    pub fn new(size: usize) -> Self {
        Self { data: vec![T::default(); size] }
    }

    pub fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    pub fn assign<RightExpr: EtlExpr<T>> (&mut self, rhs: RightExpr) {
        for i in 0..self.size() {
            self.data[i] = rhs.at(i)
        }
    }

    pub fn iter(&self) -> VectorIterator<T> {
        VectorIterator::<T> {
            vector: self,
            index: 0
        }
    }

    // Writing my own mutable iterator requires unsafe code (which I should do later)
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        return self.data.iter_mut();
    }
}

impl<T: EtlValueType> EtlExpr<T> for Vector<T> {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn at(&self, i: usize) -> T {
        self.data[i]
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

// The declaration of VectorIterator<T>

pub struct VectorIterator<'a, T: EtlValueType> {
    vector: &'a Vector<T>,
    index: usize
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

// TODO.1 Ideally, we should be able to declare that for the trait directly
impl<'a, T, RightExpr> Add<&'a RightExpr> for &'a Vector<T> where RightExpr: EtlExpr<T>, T: EtlValueType + Add<Output = T> {
    type Output = AddExpr<'a, Vector<T>, RightExpr, T>;

    fn add(self, other: &'a RightExpr) -> Self::Output {
        Self::Output::new(self, other)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let vec: Vector<i64> = Vector::<i64>::new(8);
        assert_eq!(vec.size() , 8)
    }

    #[test]
    fn construct_f64() {
        let vec: Vector<f64> = Vector::<f64>::new(1023);
        assert_eq!(vec.size() , 1023)
    }

    #[test]
    fn at() {
        let mut vec: Vector<i64> = Vector::<i64>::new(3);

        vec[0] = 9;
        vec[1] = 3;
        vec[2] = 7;

        assert_eq!(vec.at(0), 9);
        assert_eq!(vec.at(1), 3);
        assert_eq!(vec.at(2), 7);

        *vec.at_mut(1) = 77;
        assert_eq!(vec.at(1), 77);
    }
}
