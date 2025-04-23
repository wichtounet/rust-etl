use crate::base_traits::Constants;
use crate::etl_expr::*;

use std::{fmt, ops::BitOrAssign};

use rand::Rng;
use rand_distr::*;

// The declaration of Matrix2d<T>

pub struct Matrix2d<T: EtlValueType> {
    pub data: Vec<T>,
    rows: usize,
    columns: usize,
}

// The functions of Matrix2d<T>

impl<T: EtlValueType> Matrix2d<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            data: vec![T::default(); padded_size(rows * columns)],
            rows,
            columns,
        }
    }

    pub fn new_copy(rhs: &Matrix2d<T>) -> Self {
        Self {
            data: rhs.data.clone(),
            rows: rhs.rows(),
            columns: rhs.columns(),
        }
    }

    pub fn new_rand(rows: usize, columns: usize) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut mat = Self::new(rows, columns);
        mat.rand_fill();
        mat
    }

    pub fn new_rand_normal(rows: usize, columns: usize) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(rows, columns);
        mat.rand_fill_normal();
        mat
    }

    pub fn new_rand_normal_ms(rows: usize, columns: usize, mean: T, stddev: T) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(rows, columns);
        mat.rand_fill_normal_ms(mean, stddev);
        mat
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

    pub fn rand_fill_normal(&mut self)
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut rng = rand::rng();
        let n = <T as Constants>::zero();
        let p = <T as Constants>::one();
        let normal = Normal::new(n, p).unwrap();

        for i in 0..self.size() {
            self.data[i] = normal.sample(&mut rng);
        }
    }

    pub fn rand_fill_normal_ms(&mut self, mean: T, stddev: T)
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, stddev).unwrap();

        for i in 0..self.size() {
            self.data[i] = normal.sample(&mut rng);
        }
    }
}

impl<T: EtlValueType> EtlExpr<T> for Matrix2d<T> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Value;

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at2(&self, row: usize, column: usize) -> T {
        if row >= self.rows {
            panic!("Row {} is out of bounds!", row);
        }

        if column >= self.columns {
            panic!("Column {} is out of bounds!", column);
        }

        self.data[row * self.columns + column]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }
}

impl<T: EtlValueType> EtlExpr<T> for &Matrix2d<T> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Value;

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at2(&self, row: usize, column: usize) -> T {
        if row >= self.rows {
            panic!("Row {} is out of bounds!", row);
        }

        if column >= self.columns {
            panic!("Column {} is out of bounds!", column);
        }

        self.data[row * self.columns + column]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }
}

// Matrix2d<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable<T> for &'a Matrix2d<T> {
    type WrappedAs = &'a Matrix2d<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Matrix2d<T> computes as itself
impl<'a, T: EtlValueType> EtlComputable<T> for &'a Matrix2d<T> {
    fn to_data(&self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T: EtlValueType> fmt::Display for Matrix2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for row in 0..self.rows() {
            if row > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for column in 0..self.columns() {
                if column > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", self.at2(row, column))?;
            }

            write!(f, "]")?
        }

        write!(f, "]")
    }
}

// Operator overloading for Matrix2d<T>

impl<T: EtlValueType> std::ops::Index<usize> for Matrix2d<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: EtlValueType> std::ops::IndexMut<usize> for Matrix2d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// Since we can't overload Assign, we settle for BitOrAssign
impl<T: EtlValueType, RightExpr: EtlExpr<T>> BitOrAssign<RightExpr> for Matrix2d<T> {
    fn bitor_assign(&mut self, rhs: RightExpr) {
        validate_assign(self, &rhs);
        assign_direct(&mut self.data, &rhs);
    }
}

// Operations

crate::impl_add_op_value!(Matrix2d<T>);
crate::impl_sub_op_value!(Matrix2d<T>);
crate::impl_mul_op_value!(Matrix2d<T>);
crate::impl_div_op_value!(Matrix2d<T>);
crate::impl_scale_op_value!(Matrix2d<T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let mat = Matrix2d::<i64>::new(4, 2);
        assert_eq!(mat.size(), 8);
        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.columns(), 2);
    }

    #[test]
    fn construct_f64() {
        let mat = Matrix2d::<f64>::new(8, 12);
        assert_eq!(mat.size(), 96)
    }

    #[test]
    fn default_value() {
        let mat = Matrix2d::<f64>::new(8, 12);

        assert_eq!(mat.at2(0, 0), 0.0);
        assert_eq!(mat.at2(1, 1), 0.0);
        assert_eq!(mat.at2(2, 1), 0.0);
    }

    #[test]
    fn at2() {
        let mut mat = Matrix2d::<i64>::new(4, 2);

        *mat.at_mut(0, 0) = 9;
        *mat.at_mut(1, 1) = 3;

        assert_eq!(mat.at2(0, 0), 9);
        assert_eq!(mat.at2(1, 1), 3);
        assert_eq!(mat.at2(2, 1), 0);
    }

    #[test]
    fn print() {
        let mut mat = Matrix2d::<i32>::new(3, 2);

        mat[0] = 3;
        mat[1] = 2;
        mat[2] = 1;
        mat[3] = 5;
        mat[4] = 6;
        mat[5] = 9;

        println!("Display matrix: {}", mat);
        let str = format!("{mat}");
        assert_eq!(str, "[[3,2]\n[1,5]\n[6,9]]");
    }

    #[test]
    fn fill() {
        let mut mat = Matrix2d::<i64>::new(4, 2);
        mat.fill(9);

        assert_eq!(mat.at2(0, 0), 9);
        assert_eq!(mat.at2(1, 1), 9);
        assert_eq!(mat.at2(2, 1), 9);
    }

    #[test]
    fn clear() {
        let mut mat = Matrix2d::<i64>::new(4, 2);
        mat.fill(9);

        assert_eq!(mat.at2(0, 0), 9);
        assert_eq!(mat.at2(1, 1), 9);
        assert_eq!(mat.at2(2, 1), 9);

        mat.clear();

        assert_eq!(mat.at2(0, 0), 0);
        assert_eq!(mat.at2(1, 1), 0);
        assert_eq!(mat.at2(2, 1), 0);
    }

    #[test]
    fn row_major() {
        let mut mat = Matrix2d::<i64>::new(2, 2);

        mat[0] = 1;
        mat[1] = 2;
        mat[2] = 3;
        mat[3] = 4;

        assert_eq!(mat.at2(0, 0), 1);
        assert_eq!(mat.at2(0, 1), 2);
        assert_eq!(mat.at2(1, 0), 3);
        assert_eq!(mat.at2(1, 1), 4);
    }

    #[test]
    fn compound() {
        let mut a = Matrix2d::<i64>::new(2, 2);
        let mut b = Matrix2d::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        b[0] = 2;
        b[1] = 4;
        b[2] = 16;
        b[3] = 42;

        a += b;

        assert_eq!(a.at2(0, 0), 5);
        assert_eq!(a.at2(0, 1), 13);
        assert_eq!(a.at2(1, 0), 43);
        assert_eq!(a.at2(1, 1), 84);

        assert_eq!(a[0], 5);
        assert_eq!(a[1], 13);
        assert_eq!(a[2], 43);
        assert_eq!(a[3], 84);
    }

    #[test]
    fn copy() {
        let mut a = Matrix2d::<i64>::new(2, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        let b = Matrix2d::<i64>::new_copy(&a);

        assert_eq!(b.at2(0, 0), 3);
        assert_eq!(b.at2(0, 1), 9);
        assert_eq!(b.at2(1, 0), 27);
        assert_eq!(b.at2(1, 1), 42);
    }
}
