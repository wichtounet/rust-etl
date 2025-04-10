use crate::etl_expr::*;

// The declaration of BiasAddExpr

/// Expression representing the batched addition of biases to a matrix
/// LeftExpr is a vector expression
/// RightExpr is a matrix expression
/// BiasAddExpr is a vector expression
pub struct BiasAddExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of BiasAddExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> BiasAddExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            if lhs.columns() != rhs.rows() {
                panic!("Invalid bias_add dimensions ([{},{}]*[{}])", lhs.rows(), lhs.columns(), rhs.rows());
            }
        } else {
            panic!("Invalid bias_add dimensions ({}D*{}D)", LeftExpr::DIMENSIONS, RightExpr::DIMENSIONS);
        }

        let mut expr = Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_bias_add_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_bias_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            output[..self.temp.len()].copy_from_slice(&self.temp[..]);
            return;
        }

        self.compute_bias_add_impl(output);
    }

    fn compute_bias_add_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] += self.temp[n];
            }
            return;
        }

        self.compute_bias_add_impl(output);
    }

    fn compute_bias_add_sub(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] -= self.temp[n];
            }
            return;
        }

        self.compute_bias_add_impl(output);
    }

    fn compute_bias_add_scale(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] *= self.temp[n];
            }
            return;
        }

        self.compute_bias_add_impl(output);
    }

    fn compute_bias_add_impl(&self, output: &mut Vec<T>) {
        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            let m = self.lhs.value.rows();
            let n = self.lhs.value.columns();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                for row in 0..m {
                    for column in 0..n {
                        out[row * n + column] = lhs[row * n + column] + rhs[column];
                    }
                }
            };

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_bias_add<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        if OutputExpr::DIMENSIONS != 2 {
            panic!("The output of bias_add must be a 2D Matrix");
        }

        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            if lhs.rows() != self.lhs.value.rows() || lhs.columns() != self.lhs.value.columns() {
                panic!("Invalid dimensions for assignment of bias_add result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// BiasAddExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for BiasAddExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Smart;

    fn size(&self) -> usize {
        self.lhs.value.size()
    }

    fn rows(&self) -> usize {
        self.lhs.value.rows()
    }

    fn columns(&self) -> usize {
        self.lhs.value.columns()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        self.validate_bias_add(lhs);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_bias_add(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_bias_add_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_bias_add_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_bias_add_scale(output);
    }

    fn at(&self, i: usize) -> T {
        self.temp[i]
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.temp[row * self.columns() + column]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.temp
    }
}

// BiasAddExpr is an EtlWrappable
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for BiasAddExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = BiasAddExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// BiasAddExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for BiasAddExpr<T, LeftExpr, RightExpr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn bias_add<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>(
    lhs: LeftExpr,
    rhs: RightExpr,
) -> BiasAddExpr<T, LeftExpr, RightExpr> {
    BiasAddExpr::<T, LeftExpr, RightExpr>::new(lhs, rhs)
}

crate::impl_add_op_binary_expr!(BiasAddExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(BiasAddExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(BiasAddExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(BiasAddExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::bias_add_expr::bias_add;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn bias_add_simple() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Matrix2d::<i64>::new(3, 2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;

        c |= bias_add(&a, &b);

        assert_eq!(c.at2(0, 0), 8);
        assert_eq!(c.at2(0, 1), 10);
        assert_eq!(c.at2(1, 0), 10);
        assert_eq!(c.at2(1, 1), 12);
        assert_eq!(c.at2(2, 0), 12);
        assert_eq!(c.at2(2, 1), 14);
    }

    #[test]
    fn bias_add_deep() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut b = Vector::<i64>::new(2);
        let mut c = Matrix2d::<i64>::new(3, 2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;

        c |= bias_add(&a, &b) >> bias_add(&a, &b);

        assert_eq!(c.at2(0, 0), 8 * 8);
        assert_eq!(c.at2(0, 1), 10 * 10);
        assert_eq!(c.at2(1, 0), 10 * 10);
        assert_eq!(c.at2(1, 1), 12 * 12);
        assert_eq!(c.at2(2, 0), 12 * 12);
        assert_eq!(c.at2(2, 1), 14 * 14);
    }
}
