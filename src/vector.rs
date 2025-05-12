use crate::base_traits::Constants;
use crate::etl_expr::*;

use std::{fmt, ops::BitOrAssign};

use rand::Rng;
use rand_distr::*;

// The declaration of Vector<T>

pub struct Vector<T: EtlValueType> {
    pub data: Vec<T>,
    size: usize,
}

// The functions of Vector<T>

impl<T: EtlValueType> Vector<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![T::default(); padded_size(size)],
            size,
        }
    }

    pub fn new_from_expr<Expr: EtlExpr<T>>(expr: &Expr) -> Self {
        let mut vec = Self {
            data: vec![T::default(); padded_size(expr.size())],
            size: expr.size(),
        };

        for (i, value) in vec.data.iter_mut().enumerate() {
            *value = expr.at(i);
        }

        vec
    }

    pub fn new_rand(size: usize) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let mut vec = Self::new(size);
        vec.rand_fill();
        vec
    }

    pub fn new_rand_normal(size: usize) -> Self
    where
        StandardNormal: Distribution<T>,
        T: EtlValueType + rand_distr::num_traits::Float,
    {
        let mut vec = Self::new(size);
        vec.rand_fill_normal();
        vec
    }

    pub fn at_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    pub fn clear(&mut self) {
        self.data.fill(T::default());
    }

    pub fn fill(&mut self, constant: T) {
        self.data.fill(constant);
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

    pub fn inplace_axpy<RightExpr: EtlExpr<T>>(&mut self, alpha: T, beta: T, y: RightExpr) {
        validate_assign(self, &y);
        axpy_direct(&mut self.data, alpha, beta, &y);
    }

    pub fn direct_iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn direct_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: EtlValueType> Clone for Vector<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            size: self.size,
        }
    }
}

impl<T: EtlValueType> EtlExpr<T> for Vector<T> {
    const DIMENSIONS: usize = 1;
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
        self.size
    }

    fn rows(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }
}

impl<T: EtlValueType> EtlExpr<T> for &Vector<T> {
    const DIMENSIONS: usize = 1;
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
        self.size
    }

    fn rows(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn at(&self, i: usize) -> T {
        self.data[i]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }
}

// Vector<T> wraps as reference
impl<'a, T: EtlValueType> EtlWrappable<T> for &'a Vector<T> {
    type WrappedAs = &'a Vector<T>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// Vector<T> computes as itself
impl<'a, T: EtlValueType> EtlComputable<T> for &'a Vector<T> {
    fn to_data(&self) -> Vec<T> {
        self.data.clone()
    }
}

impl<T: EtlValueType> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for i in 0..self.size() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{:.6}", self.data[i])?;
        }

        write!(f, "]")
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
    fn bitor_assign(&mut self, rhs: RightExpr) {
        validate_assign(self, &rhs);
        assign_direct(&mut self.data, &rhs);
    }
}

// Operations

crate::impl_add_op_value!(Vector<T>);
crate::impl_sub_op_value!(Vector<T>);
crate::impl_mul_op_value!(Vector<T>);
crate::impl_div_op_value!(Vector<T>);
crate::impl_scale_op_value!(Vector<T>);

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
        let vec = Vector::<f64>::new(8);

        assert_eq!(vec.at(0), 0.0);
        assert_eq!(vec.at(1), 0.0);
        assert_eq!(vec.at(2), 0.0);
    }

    #[test]
    fn normal() {
        let _vec = Vector::<f64>::new_rand_normal(3);
    }

    #[test]
    fn print() {
        let mut vec = Vector::<i32>::new(3);

        vec[0] = 3;
        vec[1] = 2;
        vec[2] = 1;

        println!("Display vector: {vec}");
        let str = format!("{vec}");
        assert_eq!(str, "[3,2,1]")
    }

    #[test]
    fn clone() {
        let mut vec = Vector::<i32>::new(3);

        vec[0] = 3;
        vec[1] = 2;
        vec[2] = 1;

        let copy = vec.clone();

        assert_eq!(copy.size(), 3);
        assert_eq!(copy.at(0), 3);
        assert_eq!(copy.at(1), 2);
        assert_eq!(copy.at(2), 1);
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
