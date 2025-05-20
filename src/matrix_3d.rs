use crate::base_traits::Constants;
use crate::etl_expr::*;

use std::{fmt, ops::BitOrAssign};

use rand::Rng;
use rand_distr::*;

// The declaration of Matrix3d<T>

#[derive(Clone)]
pub struct Matrix3d<T: EtlValueType> {
    pub data: Vec<T>,
    m: usize,
    n: usize,
    k: usize,
}

// The functions of Matrix3d<T>

impl<T: EtlValueType> Matrix3d<T> {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            data: vec![T::default(); padded_size(m * n * k)],
            m,
            n,
            k,
        }
    }

    pub fn new_from_expr<Expr: EtlExpr<T>>(expr: Expr) -> Self {
        assert_eq!(Expr::DIMENSIONS, 3);

        let mut vec = Self {
            data: vec![T::default(); padded_size(expr.size())],
            m: expr.dim(0),
            n: expr.dim(1),
            k: expr.dim(2),
        };

        for (lhs, rhs) in vec.data.iter_mut().zip(expr.iter()) {
            *lhs = rhs;
        }

        vec
    }

    pub fn new_copy(rhs: &Matrix3d<T>) -> Self {
        Self {
            data: rhs.data.clone(),
            m: rhs.m,
            n: rhs.n,
            k: rhs.k,
        }
    }

    pub fn new_iota(m: usize, n: usize, k: usize, value: T) -> Self {
        let mut mat = Self {
            data: vec![T::default(); padded_size(m * n * k)],
            m,
            n,
            k,
        };
        mat.iota_fill(value);
        mat
    }

    pub fn new_rand(m: usize, n: usize, k: usize) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut mat = Self::new(m, n, k);
        mat.rand_fill();
        mat
    }

    pub fn new_rand_normal(m: usize, n: usize, k: usize) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(m, n, k);
        mat.rand_fill_normal();
        mat
    }

    pub fn new_rand_normal_ms(m: usize, n: usize, k: usize, mean: T, stddev: T) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(m, n, k);
        mat.rand_fill_normal_ms(mean, stddev);
        mat
    }

    pub fn at3_mut(&mut self, m: usize, n: usize, k: usize) -> &mut T {
        if m >= self.m {
            panic!("Row {m} is out of bounds!");
        }

        if n >= self.n {
            panic!("Column {n} is out of bounds!");
        }

        if k >= self.k {
            panic!("Third dimension {k} is out of bounds!");
        }

        &mut self.data[m * self.n * self.k + n * self.k + k]
    }

    pub fn clear(&mut self) {
        self.data.fill(T::default());
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    pub fn iota_fill(&mut self, value: T) {
        let mut acc = value;
        for value in self.data.iter_mut() {
            *value = acc;
            acc += T::one();
        }
    }

    pub fn rand_fill(&mut self)
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut rng = rand::rng();

        for value in self.data.iter_mut() {
            *value = rng.random::<T>();
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

        for value in self.data.iter_mut() {
            *value = normal.sample(&mut rng);
        }
    }

    pub fn rand_fill_normal_ms(&mut self, mean: T, stddev: T)
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut rng = rand::rng();
        let normal = Normal::new(mean, stddev).unwrap();

        for value in self.data.iter_mut() {
            *value = normal.sample(&mut rng);
        }
    }

    pub fn inplace_axpy<RightExpr: EtlExpr<T>>(&mut self, alpha: T, beta: T, y: RightExpr) {
        validate_assign(self, &y);
        axpy_direct(&mut self.data, alpha, beta, &y);
    }
}

impl<T: EtlValueType> EtlExpr<T> for Matrix3d<T> {
    const DIMENSIONS: usize = 3;
    const TYPE: EtlType = EtlType::Value;
    const THREAD_SAFE: bool = true;

    type Iter<'x>
        = std::iter::Cloned<std::slice::Iter<'x, T>>
    where
        T: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter().cloned()
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        self.data[range].iter().cloned()
    }

    fn size(&self) -> usize {
        self.m * self.n * self.k
    }

    fn rows(&self) -> usize {
        self.m
    }

    fn columns(&self) -> usize {
        self.n
    }

    fn dim(&self, i: usize) -> usize {
        match i {
            0 => self.m,
            1 => self.n,
            2 => self.k,
            _ => panic!("Invalid dimension access"),
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at3(&self, m: usize, n: usize, k: usize) -> T {
        if m >= self.m {
            panic!("Row {m} is out of bounds!");
        }

        if n >= self.n {
            panic!("Column {n} is out of bounds!");
        }

        if k >= self.k {
            panic!("Third dimension {k} is out of bounds!");
        }

        self.data[m * self.n * self.k + n * self.k + k]
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }
}

impl<T: EtlValueType> EtlExpr<T> for &Matrix3d<T> {
    const DIMENSIONS: usize = 3;
    const TYPE: EtlType = EtlType::Value;
    const THREAD_SAFE: bool = true;

    type Iter<'x>
        = std::iter::Cloned<std::slice::Iter<'x, T>>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter().cloned()
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        self.data[range].iter().cloned()
    }

    fn size(&self) -> usize {
        self.m * self.n * self.k
    }

    fn rows(&self) -> usize {
        self.m
    }

    fn columns(&self) -> usize {
        self.n
    }

    fn dim(&self, i: usize) -> usize {
        match i {
            0 => self.m,
            1 => self.n,
            2 => self.k,
            _ => panic!("Invalid dimension access"),
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at3(&self, m: usize, n: usize, k: usize) -> T {
        if m >= self.m {
            panic!("Row {m} is out of bounds!");
        }

        if n >= self.n {
            panic!("Column {n} is out of bounds!");
        }

        if k >= self.k {
            panic!("Third dimension {k} is out of bounds!");
        }

        self.data[m * self.n * self.k + n * self.k + k]
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }
}

// Matrix3d<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable<T> for &'a Matrix3d<T> {
    type WrappedAs = &'a Matrix3d<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Matrix3d<T> computes as itself
impl<T: EtlValueType> EtlComputable<T> for &Matrix3d<T> {
    fn to_data(&self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T: EtlValueType> fmt::Display for Matrix3d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for row in 0..self.m {
            if row > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for column in 0..self.n {
                if column > 0 {
                    writeln!(f)?;
                }
                write!(f, "[")?;
                for inner in 0..self.k {
                    if inner > 0 {
                        write!(f, ",")?;
                    }

                    write!(f, "{:.6}", self.at3(row, column, inner))?;
                }
                write!(f, "]")?
            }

            write!(f, "]")?
        }

        write!(f, "]")
    }
}

// Operator overloading for Matrix3d<T>

impl<T: EtlValueType> std::ops::Index<usize> for Matrix3d<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: EtlValueType> std::ops::IndexMut<usize> for Matrix3d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// Since we can't overload Assign, we settle for BitOrAssign
impl<T: EtlValueType, RightExpr: EtlExpr<T>> BitOrAssign<RightExpr> for Matrix3d<T> {
    fn bitor_assign(&mut self, rhs: RightExpr) {
        validate_assign(self, &rhs);
        assign_direct(&mut self.data, &rhs);
    }
}

// Operations

crate::impl_add_op_value!(Matrix3d<T>);
crate::impl_sub_op_value!(Matrix3d<T>);
crate::impl_mul_op_value!(Matrix3d<T>);
crate::impl_div_op_value!(Matrix3d<T>);
crate::impl_scale_op_value!(Matrix3d<T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let mat = Matrix3d::<i64>::new(4, 2, 3);
        assert_eq!(mat.size(), 24);
        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.columns(), 2);
        assert_eq!(mat.dim(0), 4);
        assert_eq!(mat.dim(1), 2);
        assert_eq!(mat.dim(2), 3);
    }

    #[test]
    fn construct_f64() {
        let mat = Matrix3d::<f64>::new(8, 12, 2);
        assert_eq!(mat.size(), 192)
    }

    #[test]
    fn default_value() {
        let mat = Matrix3d::<f64>::new(8, 12, 3);

        assert_eq!(mat.at3(0, 0, 0), 0.0);
        assert_eq!(mat.at3(1, 1, 2), 0.0);
        assert_eq!(mat.at3(2, 1, 1), 0.0);
    }

    #[test]
    fn at2() {
        let mut mat = Matrix3d::<i64>::new(4, 2, 4);

        *mat.at3_mut(0, 0, 0) = 9;
        *mat.at3_mut(1, 1, 1) = 3;

        assert_eq!(mat.at3(0, 0, 0), 9);
        assert_eq!(mat.at3(1, 1, 1), 3);
        assert_eq!(mat.at3(2, 1, 1), 0);
    }

    #[test]
    fn print() {
        let mut mat = Matrix3d::<i32>::new(3, 2, 2);

        mat.iota_fill(1);

        println!("Display matrix: {mat}");
        let str = format!("{mat}");
        assert_eq!(str, "[[[1,2]\n[3,4]]\n[[5,6]\n[7,8]]\n[[9,10]\n[11,12]]]");
    }

    #[test]
    fn print_float() {
        let mut mat = Matrix3d::<f32>::new(3, 2, 2);

        mat.iota_fill(1.0);

        println!("Display matrix: {mat}");
        let str = format!("{mat}");
        assert_eq!(
            str,
            "[[[1.000000,2.000000]\n[3.000000,4.000000]]\n[[5.000000,6.000000]\n[7.000000,8.000000]]\n[[9.000000,10.000000]\n[11.000000,12.000000]]]"
        );
    }

    #[test]
    fn clone() {
        let mut vec = Matrix3d::<i32>::new(2, 2, 2);

        vec[0] = 3;
        vec[1] = 2;
        vec[2] = 1;
        vec[3] = 9;

        let copy = vec.clone();

        assert_eq!(copy.size(), 8);
        assert_eq!(copy.rows(), 2);
        assert_eq!(copy.columns(), 2);
        assert_eq!(copy.at(0), 3);
        assert_eq!(copy.at(1), 2);
        assert_eq!(copy.at(2), 1);
        assert_eq!(copy.at(3), 9);
    }

    #[test]
    fn fill() {
        let mut mat = Matrix3d::<i64>::new(4, 2, 3);
        mat.fill(9);

        assert_eq!(mat.at3(0, 0, 1), 9);
        assert_eq!(mat.at3(1, 1, 1), 9);
        assert_eq!(mat.at3(2, 1, 1), 9);
    }

    #[test]
    fn clear() {
        let mut mat = Matrix3d::<i64>::new(4, 2, 3);
        mat.fill(9);

        assert_eq!(mat.at3(0, 0, 0), 9);
        assert_eq!(mat.at3(1, 1, 0), 9);
        assert_eq!(mat.at3(2, 1, 2), 9);

        mat.clear();

        assert_eq!(mat.at3(0, 0, 0), 0);
        assert_eq!(mat.at3(1, 1, 1), 0);
        assert_eq!(mat.at3(2, 1, 2), 0);
    }

    #[test]
    fn row_major() {
        let mut mat = Matrix3d::<i64>::new(2, 2, 3);
        mat.iota_fill(1);

        assert_eq!(mat.at3(0, 0, 0), 1);
        assert_eq!(mat.at3(0, 0, 1), 2);
        assert_eq!(mat.at3(0, 0, 2), 3);
        assert_eq!(mat.at3(0, 1, 0), 4);
        assert_eq!(mat.at3(0, 1, 1), 5);
        assert_eq!(mat.at3(0, 1, 2), 6);
        assert_eq!(mat.at3(1, 0, 0), 7);
        assert_eq!(mat.at3(1, 0, 1), 8);
        assert_eq!(mat.at3(1, 0, 2), 9);
        assert_eq!(mat.at3(1, 1, 0), 10);
        assert_eq!(mat.at3(1, 1, 1), 11);
        assert_eq!(mat.at3(1, 1, 2), 12);
    }

    #[test]
    fn compound() {
        let mut a = Matrix3d::<i64>::new(2, 2, 3);
        let mut b = Matrix3d::<i64>::new(2, 2, 3);

        a.iota_fill(1);
        b.iota_fill(2);

        a += b;

        assert_eq!(a.at3(0, 0, 0), 1 + 2);
        assert_eq!(a.at3(0, 0, 1), 2 + 3);
        assert_eq!(a.at3(0, 0, 2), 3 + 4);
        assert_eq!(a.at3(0, 1, 0), 4 + 5);
        assert_eq!(a.at3(0, 1, 1), 5 + 6);
        assert_eq!(a.at3(0, 1, 2), 6 + 7);
        assert_eq!(a.at3(1, 0, 0), 7 + 8);
        assert_eq!(a.at3(1, 0, 1), 8 + 9);
        assert_eq!(a.at3(1, 0, 2), 9 + 10);
        assert_eq!(a.at3(1, 1, 0), 10 + 11);
        assert_eq!(a.at3(1, 1, 1), 11 + 12);
        assert_eq!(a.at3(1, 1, 2), 12 + 13);
    }

    #[test]
    fn inplace_axpy() {
        let mut a = Matrix3d::<i64>::new(2, 2, 3);
        let mut b = Matrix3d::<i64>::new(2, 2, 3);

        a.iota_fill(1);
        b.iota_fill(2);

        a.inplace_axpy(3, 5, b);

        for (n, it) in a.iter().enumerate() {
            assert_eq!(it, (3 * (n + 1) + 5 * (n + 2)).try_into().unwrap());
        }
    }

    #[test]
    fn copy() {
        let mut a = Matrix3d::<i64>::new(2, 2, 3);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        let b = Matrix3d::<i64>::new_copy(&a);

        assert_eq!(b.at3(0, 0, 0), 3);
        assert_eq!(b.at3(0, 0, 1), 9);
        assert_eq!(b.at3(0, 0, 2), 27);
        assert_eq!(b.at3(0, 1, 0), 42);
    }
}
