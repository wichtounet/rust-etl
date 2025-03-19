use crate::etl::etl_expr::*;
use crate::etl::add_expr::AddExpr;

use crate::impl_add_op_value;

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::BitOrAssign;

use rand::Rng;

// The declaration of Matrix<T>

pub struct Matrix<T: EtlValueType> {
    data: Vec<T>,
    rows: usize,
    columns: usize,
}

// The functions of Matrix<T>

impl<T: EtlValueType> Matrix<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self { 
            data: vec![T::default(); rows * columns],
            rows: rows,
            columns: columns
        }
    }

    pub fn new_rand(rows: usize, columns: usize) -> Self
    where rand::distr::StandardUniform: rand::distr::Distribution<T> {
        let mut mat = Self::new(rows, columns);
        mat.rand_fill();
        mat
    }

    pub fn at(&self, row: usize, column: usize) -> T {
        if row >= self.rows {
            panic!("Row {} is out of bounds!", row);
        }

        if column >= self.columns {
            panic!("Column {} is out of bounds!", column);
        }

        self.data[row * self.columns + column]
    }

    pub fn at_mut(&mut self, row: usize, column: usize) -> &mut T {
        if row >= self.rows {
            panic!("Row {} is out of bounds!", row);
        }

        if column >= self.columns {
            panic!("Column {} is out of bounds!", column);
        }

        &mut self.data[row * self.columns + column]
    }

    pub fn assign_direct<RightExpr: EtlExpr<Type = T>> (&mut self, rhs: RightExpr) {
        for i in 0..self.size() {
            self.data[i] = rhs.at(i);
        }
    }

    pub fn add_assign_direct<RightExpr: EtlExpr<Type = T>> (&mut self, rhs: RightExpr)
    where T: AddAssign<T> {
        for i in 0..self.size() {
            self.data[i] += rhs.at(i);
        }
    }

    pub fn rand_fill(&mut self)
    where rand::distr::StandardUniform: rand::distr::Distribution<T> {
        let mut rng = rand::rng();

        for i in 0..self.size() {
            self.data[i] = rng.random::<T>();
        }
    }
}

impl<T: EtlValueType> EtlExpr for Matrix<T> {
    type Type = T;

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    fn at(&self, i: usize) -> Self::Type {
        self.data[i]
    }
}

impl<'a, T: EtlValueType> EtlExpr for &'a Matrix<T> {
    type Type = T;

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    fn at(&self, i: usize) -> Self::Type {
        self.data[i]
    }
}

// Matrix<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable for &'a Matrix<T> {
    type WrappedAs = &'a Matrix<T>;

    fn wrap(self) -> EtlWrapper<Self::WrappedAs> {
        EtlWrapper { value: &self }
    }
}

// Operator overloading for Matrix<T>

impl<T: EtlValueType> std::ops::Index<usize> for Matrix<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: EtlValueType> std::ops::IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// Since we can't overload Assign, we settle for BitOrAssign
impl<T: EtlValueType, RightExpr: EtlExpr<Type = T>> BitOrAssign<RightExpr> for Matrix<T> {
    fn bitor_assign(&mut self, other: RightExpr) {
        self.assign_direct(other);
    }
}

// Operations

impl_add_op_value!(Matrix<T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let mat: Matrix<i64> = Matrix::<i64>::new(4, 2);
        assert_eq!(mat.size() , 8)
    }

    #[test]
    fn construct_f64() {
        let mat: Matrix<f64> = Matrix::<f64>::new(8, 12);
        assert_eq!(mat.size() , 96)
    }

    #[test]
    fn at() {
        let mut mat: Matrix<i64> = Matrix::<i64>::new(4, 2);

        *mat.at_mut(0,0) = 9;
        *mat.at_mut(1,1) = 3;

        assert_eq!(mat.at(0, 0), 9);
        assert_eq!(mat.at(1, 1), 3);
        assert_eq!(mat.at(2, 1), 0);
    }

    #[test]
    fn row_major() {
        let mut mat: Matrix<i64> = Matrix::<i64>::new(2, 2);

        mat[0] = 1;
        mat[1] = 2;
        mat[2] = 3;
        mat[3] = 4;

        assert_eq!(mat.at(0, 0), 1);
        assert_eq!(mat.at(0, 1), 2);
        assert_eq!(mat.at(1, 0), 3);
        assert_eq!(mat.at(1, 1), 4);
    }

    #[test]
    fn compound() {
        let mut a = Matrix::<i64>::new(2, 2);
        let mut b = Matrix::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        b[0] = 2;
        b[1] = 4;
        b[2] = 16;
        b[3] = 42;

        a += b;

        assert_eq!(a.at(0, 0), 5);
        assert_eq!(a.at(0, 1), 13);
        assert_eq!(a.at(1, 0), 43);
        assert_eq!(a.at(1, 1), 84);

        assert_eq!(a[0], 5);
        assert_eq!(a[1], 13);
        assert_eq!(a[2], 43);
        assert_eq!(a[3], 84);
    }
}
