use crate::base_traits::Constants;
use crate::etl_expr::*;

use std::{fmt, ops::BitOrAssign};

use rand::Rng;
use rand_distr::*;

// The declaration of Matrix4d<T>

#[derive(Clone)]
pub struct Matrix4d<T: EtlValueType> {
    pub data: Vec<T>,
    b: usize,
    c: usize,
    w: usize,
    h: usize,
}

// The functions of Matrix4d<T>

impl<T: EtlValueType> Matrix4d<T> {
    pub fn new(b: usize, c: usize, w: usize, h: usize) -> Self {
        Self {
            data: vec![T::default(); padded_size(b * c * w * h)],
            b,
            c,
            w,
            h,
        }
    }

    pub fn new_from_expr<Expr: EtlExpr<T>>(expr: Expr) -> Self {
        assert_eq!(Expr::DIMENSIONS, 3);

        let mut vec = Self {
            data: vec![T::default(); padded_size(expr.size())],
            b: expr.dim(0),
            c: expr.dim(1),
            w: expr.dim(2),
            h: expr.dim(3),
        };

        for (lhs, rhs) in vec.data.iter_mut().zip(expr.iter()) {
            *lhs = rhs;
        }

        vec
    }

    pub fn new_copy(rhs: &Matrix4d<T>) -> Self {
        Self {
            data: rhs.data.clone(),
            b: rhs.b,
            c: rhs.c,
            w: rhs.w,
            h: rhs.h,
        }
    }

    pub fn new_iota(b: usize, c: usize, w: usize, h: usize, value: T) -> Self {
        let mut mat = Self {
            data: vec![T::default(); padded_size(b * c * w * h)],
            b,
            c,
            w,
            h,
        };
        mat.iota_fill(value);
        mat
    }

    pub fn new_rand(b: usize, c: usize, w: usize, h: usize) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut mat = Self::new(b, c, w, h);
        mat.rand_fill();
        mat
    }

    pub fn new_rand_normal(b: usize, c: usize, w: usize, h: usize) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(b, c, w, h);
        mat.rand_fill_normal();
        mat
    }

    pub fn new_rand_normal_ms(b: usize, c: usize, w: usize, h: usize, mean: T, stddev: T) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut mat = Self::new(b, c, w, h);
        mat.rand_fill_normal_ms(mean, stddev);
        mat
    }

    pub fn at4_mut(&mut self, b: usize, c: usize, w: usize, h: usize) -> &mut T {
        if b >= self.b {
            panic!("Row {b} is out of bounds!");
        }

        if c >= self.c {
            panic!("Column {c} is out of bounds!");
        }

        if w >= self.w {
            panic!("Third dimension {w} is out of bounds!");
        }

        if h >= self.h {
            panic!("Fourth dimension {h} is out of bounds!");
        }

        &mut self.data[b * self.c * self.w * self.h + c * self.w * self.h + w * self.h + h]
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

impl<T: EtlValueType> EtlExpr<T> for Matrix4d<T> {
    const DIMENSIONS: usize = 4;
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
        self.b * self.c * self.w * self.h
    }

    fn rows(&self) -> usize {
        self.b
    }

    fn columns(&self) -> usize {
        self.c
    }

    fn dim(&self, i: usize) -> usize {
        match i {
            0 => self.b,
            1 => self.c,
            2 => self.w,
            3 => self.h,
            _ => panic!("Invalid dimension access"),
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at4(&self, b: usize, c: usize, w: usize, h: usize) -> T {
        if b >= self.b {
            panic!("Row {b} is out of bounds!");
        }

        if c >= self.c {
            panic!("Column {c} is out of bounds!");
        }

        if w >= self.w {
            panic!("Third dimension {w} is out of bounds!");
        }

        if h >= self.h {
            panic!("Fourth dimension {h} is out of bounds!");
        }

        self.data[b * self.c * self.w * self.h + c * self.w * self.h + w * self.h + h]
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }
}

impl<T: EtlValueType> EtlExpr<T> for &Matrix4d<T> {
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
        self.b * self.c * self.w * self.h
    }

    fn rows(&self) -> usize {
        self.b
    }

    fn columns(&self) -> usize {
        self.c
    }

    fn dim(&self, i: usize) -> usize {
        match i {
            0 => self.b,
            1 => self.c,
            2 => self.w,
            3 => self.h,
            _ => panic!("Invalid dimension access"),
        }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn at4(&self, b: usize, c: usize, w: usize, h: usize) -> T {
        if b >= self.b {
            panic!("Row {b} is out of bounds!");
        }

        if c >= self.c {
            panic!("Column {c} is out of bounds!");
        }

        if w >= self.w {
            panic!("Third dimension {w} is out of bounds!");
        }

        if h >= self.h {
            panic!("Fourth dimension {h} is out of bounds!");
        }

        self.data[b * self.c * self.w * self.h + c * self.w * self.h + w * self.h + h]
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }
}

// Matrix4d<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable<T> for &'a Matrix4d<T> {
    type WrappedAs = &'a Matrix4d<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Matrix4d<T> computes as itself
impl<T: EtlValueType> EtlComputable<T> for &Matrix4d<T> {
    fn to_data(&self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T: EtlValueType> fmt::Display for Matrix4d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for b in 0..self.b {
            if b > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for row in 0..self.c {
                if row > 0 {
                    writeln!(f)?;
                }
                write!(f, "[")?;
                for column in 0..self.w {
                    if column > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "[")?;
                    for inner in 0..self.h {
                        if inner > 0 {
                            write!(f, ",")?;
                        }

                        write!(f, "{:.6}", self.at4(b, row, column, inner))?;
                    }
                    write!(f, "]")?
                }

                write!(f, "]")?
            }

            write!(f, "]")?
        }

        write!(f, "]")
    }
}

// Operator overloading for Matrix4d<T>

impl<T: EtlValueType> std::ops::Index<usize> for Matrix4d<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: EtlValueType> std::ops::IndexMut<usize> for Matrix4d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

// Since we can't overload Assign, we settle for BitOrAssign
impl<T: EtlValueType, RightExpr: EtlExpr<T>> BitOrAssign<RightExpr> for Matrix4d<T> {
    fn bitor_assign(&mut self, rhs: RightExpr) {
        validate_assign(self, &rhs);
        assign_direct(&mut self.data, &rhs);
    }
}

// Operations

crate::impl_add_op_value!(Matrix4d<T>);
crate::impl_sub_op_value!(Matrix4d<T>);
crate::impl_mul_op_value!(Matrix4d<T>);
crate::impl_div_op_value!(Matrix4d<T>);
crate::impl_scale_op_value!(Matrix4d<T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_i64() {
        let mat = Matrix4d::<i64>::new(4, 2, 3, 2);
        assert_eq!(mat.size(), 48);
        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.columns(), 2);
        assert_eq!(mat.dim(0), 4);
        assert_eq!(mat.dim(1), 2);
        assert_eq!(mat.dim(2), 3);
        assert_eq!(mat.dim(3), 2);
    }

    #[test]
    fn construct_f64() {
        let mat = Matrix4d::<f64>::new(8, 12, 2, 3);
        assert_eq!(mat.size(), 576)
    }

    #[test]
    fn default_value() {
        let mat = Matrix4d::<f64>::new(8, 12, 3, 3);

        assert_eq!(mat.at4(0, 0, 0, 0), 0.0);
        assert_eq!(mat.at4(1, 1, 2, 1), 0.0);
        assert_eq!(mat.at4(2, 1, 1, 2), 0.0);
    }

    #[test]
    fn at4() {
        let mut mat = Matrix4d::<i64>::new(4, 2, 4, 3);

        *mat.at4_mut(0, 0, 0, 0) = 9;
        *mat.at4_mut(1, 1, 1, 1) = 3;

        assert_eq!(mat.at4(0, 0, 0, 0), 9);
        assert_eq!(mat.at4(1, 1, 1, 1), 3);
        assert_eq!(mat.at4(2, 1, 1, 0), 0);
    }

    #[test]
    fn print() {
        let mut mat = Matrix4d::<i32>::new(3, 2, 2, 2);

        mat.iota_fill(1);

        println!("Display matrix: {mat}");
        let str = format!("{mat}");
        assert_eq!(
            str,
            "[[[[1,2]\n[3,4]]\n[[5,6]\n[7,8]]]\n[[[9,10]\n[11,12]]\n[[13,14]\n[15,16]]]\n[[[17,18]\n[19,20]]\n[[21,22]\n[23,24]]]]"
        );
    }

    #[test]
    fn print_float() {
        let mut mat = Matrix4d::<f32>::new(3, 2, 2, 2);

        mat.iota_fill(1.0);

        println!("Display matrix: {mat}");
        let str = format!("{mat}");
        assert_eq!(
            str,
            "[[[[1.000000,2.000000]\n[3.000000,4.000000]]\n[[5.000000,6.000000]\n[7.000000,8.000000]]]\n[[[9.000000,10.000000]\n[11.000000,12.000000]]\n[[13.000000,14.000000]\n[15.000000,16.000000]]]\n[[[17.000000,18.000000]\n[19.000000,20.000000]]\n[[21.000000,22.000000]\n[23.000000,24.000000]]]]"
        );
    }

    #[test]
    fn clone() {
        let mut vec = Matrix4d::<i32>::new(2, 2, 2, 2);

        vec[0] = 3;
        vec[1] = 2;
        vec[2] = 1;
        vec[3] = 9;

        let copy = vec.clone();

        assert_eq!(copy.size(), 16);
        assert_eq!(copy.rows(), 2);
        assert_eq!(copy.columns(), 2);
        assert_eq!(copy.at(0), 3);
        assert_eq!(copy.at(1), 2);
        assert_eq!(copy.at(2), 1);
        assert_eq!(copy.at(3), 9);
    }

    #[test]
    fn fill() {
        let mut mat = Matrix4d::<i64>::new(4, 2, 3, 2);
        mat.fill(9);

        assert_eq!(mat.at4(0, 0, 1, 0), 9);
        assert_eq!(mat.at4(1, 1, 1, 1), 9);
        assert_eq!(mat.at4(2, 1, 1, 1), 9);
    }

    #[test]
    fn clear() {
        let mut mat = Matrix4d::<i64>::new(4, 2, 3, 2);
        mat.fill(9);

        assert_eq!(mat.at4(0, 0, 0, 1), 9);
        assert_eq!(mat.at4(1, 1, 0, 1), 9);
        assert_eq!(mat.at4(2, 1, 2, 1), 9);

        mat.clear();

        assert_eq!(mat.at4(0, 0, 0, 1), 0);
        assert_eq!(mat.at4(1, 1, 1, 1), 0);
        assert_eq!(mat.at4(2, 1, 2, 1), 0);
    }

    #[test]
    fn row_major() {
        let mut mat = Matrix4d::<i64>::new(2, 2, 3, 2);
        mat.iota_fill(1);

        assert_eq!(mat.at4(0, 0, 0, 0), 1);
        assert_eq!(mat.at4(0, 0, 0, 1), 2);
        assert_eq!(mat.at4(0, 0, 1, 0), 3);
        assert_eq!(mat.at4(0, 0, 1, 1), 4);
        assert_eq!(mat.at4(0, 0, 2, 0), 5);
        assert_eq!(mat.at4(0, 0, 2, 1), 6);
        assert_eq!(mat.at4(0, 1, 0, 0), 7);
        assert_eq!(mat.at4(0, 1, 0, 1), 8);
        assert_eq!(mat.at4(0, 1, 1, 0), 9);
        assert_eq!(mat.at4(0, 1, 1, 1), 10);
        assert_eq!(mat.at4(0, 1, 2, 0), 11);
        assert_eq!(mat.at4(0, 1, 2, 1), 12);
        assert_eq!(mat.at4(1, 0, 0, 0), 13);
        assert_eq!(mat.at4(1, 0, 0, 1), 14);
        assert_eq!(mat.at4(1, 0, 1, 0), 15);
        assert_eq!(mat.at4(1, 0, 1, 1), 16);
        assert_eq!(mat.at4(1, 0, 2, 0), 17);
        assert_eq!(mat.at4(1, 0, 2, 1), 18);
        assert_eq!(mat.at4(1, 1, 0, 0), 19);
        assert_eq!(mat.at4(1, 1, 0, 1), 20);
        assert_eq!(mat.at4(1, 1, 1, 0), 21);
        assert_eq!(mat.at4(1, 1, 1, 1), 22);
        assert_eq!(mat.at4(1, 1, 2, 0), 23);
        assert_eq!(mat.at4(1, 1, 2, 1), 24);
    }

    #[test]
    fn compound() {
        let mut a = Matrix4d::<i64>::new(2, 2, 3, 2);
        let mut b = Matrix4d::<i64>::new(2, 2, 3, 2);

        a.iota_fill(1);
        b.iota_fill(2);

        a += b;

        assert_eq!(a.at4(0, 0, 0, 0), 1 + 2);
        assert_eq!(a.at4(0, 0, 0, 1), 2 + 3);
        assert_eq!(a.at4(0, 0, 1, 0), 3 + 4);
        assert_eq!(a.at4(0, 0, 1, 1), 4 + 5);
        assert_eq!(a.at4(0, 0, 2, 0), 5 + 6);
        assert_eq!(a.at4(0, 0, 2, 1), 6 + 7);
        assert_eq!(a.at4(0, 1, 0, 0), 7 + 8);
        assert_eq!(a.at4(0, 1, 0, 1), 8 + 9);
        assert_eq!(a.at4(0, 1, 1, 0), 9 + 10);
        assert_eq!(a.at4(0, 1, 1, 1), 10 + 11);
        assert_eq!(a.at4(0, 1, 2, 0), 11 + 12);
        assert_eq!(a.at4(0, 1, 2, 1), 12 + 13);
        assert_eq!(a.at4(1, 0, 0, 0), 13 + 14);
        assert_eq!(a.at4(1, 0, 0, 1), 14 + 15);
        assert_eq!(a.at4(1, 0, 1, 0), 15 + 16);
        assert_eq!(a.at4(1, 0, 1, 1), 16 + 17);
        assert_eq!(a.at4(1, 0, 2, 0), 17 + 18);
        assert_eq!(a.at4(1, 0, 2, 1), 18 + 19);
        assert_eq!(a.at4(1, 1, 0, 0), 19 + 20);
        assert_eq!(a.at4(1, 1, 0, 1), 20 + 21);
        assert_eq!(a.at4(1, 1, 1, 0), 21 + 22);
        assert_eq!(a.at4(1, 1, 1, 1), 22 + 23);
        assert_eq!(a.at4(1, 1, 2, 0), 23 + 24);
        assert_eq!(a.at4(1, 1, 2, 1), 24 + 25);
    }

    #[test]
    fn inplace_axpy() {
        let mut a = Matrix4d::<i64>::new(2, 2, 3, 2);
        let mut b = Matrix4d::<i64>::new(2, 2, 3, 2);

        a.iota_fill(1);
        b.iota_fill(2);

        a.inplace_axpy(3, 5, b);

        for (n, it) in a.iter().enumerate() {
            assert_eq!(it, (3 * (n + 1) + 5 * (n + 2)).try_into().unwrap());
        }
    }

    #[test]
    fn copy() {
        let mut a = Matrix4d::<i64>::new(2, 2, 3, 2);

        a[0] = 3;
        a[1] = 9;
        a[2] = 27;
        a[3] = 42;

        let b = Matrix4d::<i64>::new_copy(&a);

        assert_eq!(b.at4(0, 0, 0, 0), 3);
        assert_eq!(b.at4(0, 0, 0, 1), 9);
        assert_eq!(b.at4(0, 0, 1, 0), 27);
        assert_eq!(b.at4(0, 0, 1, 1), 42);
    }
}
