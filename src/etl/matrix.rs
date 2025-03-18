use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlValueType;
use crate::etl::add_expr::AddExpr;

use std::ops::Add;

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

    pub fn assign<RightExpr> (&mut self, rhs: RightExpr)
    where RightExpr: EtlExpr<Type = T>,
    {
        for i in 0..self.size() {
            self.data[i] = rhs.at(i);
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

// Operations

// TODO.1 Ideally, we should be able to declare that for the trait directly
impl<'a, T, RightExpr> Add<&'a RightExpr> for &'a Matrix<T>
where RightExpr: EtlExpr, T: EtlValueType + Add<Output = T> {
    type Output = AddExpr<'a, Matrix<T>, RightExpr>;

    fn add(self, other: &'a RightExpr) -> Self::Output {
        Self::Output::new(self, other)
    }
}

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
}
