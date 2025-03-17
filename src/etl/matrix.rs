// The declaration of Matrix<T>

pub struct Matrix<T: Default + Clone + Copy> {
    data: Vec<T>,
    rows: usize,
    columns: usize,
}

// The functions of Matrix<T>

impl<T: Default + Clone + Copy> Matrix<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self { 
            data: vec![T::default(); rows * columns],
            rows: rows,
            columns: columns
        }
    }

    pub fn size(&self) -> usize {
        self.rows * self.columns
    }

    pub fn at(&self, row: usize, column: usize) -> T {
        self.data[row * self.columns + column]
    }

    pub fn at_mut(&mut self, row: usize, column: usize) -> &mut T {
        &mut self.data[row * self.columns + column]
    }

    // TODO: Add iterators?
}

// Operator overloading for Matrix<T>

impl<T: Default + Clone + Copy> std::ops::Index<usize> for Matrix<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Default + Clone + Copy> std::ops::IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
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
}
