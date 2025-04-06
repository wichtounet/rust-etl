use crate::etl_expr::*;

// The declaration of BiasBatchSumExpr

/// Expression representing the batched addition of biases to a matrix
pub struct BiasBatchSumExpr<T: EtlValueType, Expr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, Expr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of BiasBatchSumExpr

impl<T: EtlValueType, Expr: WrappableExpr<T>> BiasBatchSumExpr<T, Expr> {
    pub fn new(lhs: Expr) -> Self {
        if Expr::DIMENSIONS != 2 {
            panic!("Invalid bias_batch_sum dimensions ({}D)", Expr::DIMENSIONS);
        }

        let mut expr = Self {
            lhs: lhs.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_bias_batch_sum_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_bias_batch_sum(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            output[..self.temp.len()].copy_from_slice(&self.temp[..]);
            return;
        }

        self.compute_bias_batch_sum_impl(output);
    }

    fn compute_bias_batch_sum_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] += self.temp[n];
            }
            return;
        }

        self.compute_bias_batch_sum_impl(output);
    }

    fn compute_bias_batch_sum_sub(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] -= self.temp[n];
            }
            return;
        }

        self.compute_bias_batch_sum_impl(output);
    }

    fn compute_bias_batch_sum_scale(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] *= self.temp[n];
            }
            return;
        }

        self.compute_bias_batch_sum_impl(output);
    }

    fn compute_bias_batch_sum_impl(&self, output: &mut Vec<T>) {
        if Expr::DIMENSIONS == 2 {
            let b = self.lhs.value.rows();
            let m = self.lhs.value.columns();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>| {
                for batch in 0..b {
                    for row in 0..m {
                        out[row] += lhs[batch * m + row]
                    }
                }
            };

            forward_data_unary(output, &self.lhs.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_bias_batch_sum<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        if OutputExpr::DIMENSIONS != 1 {
            panic!("The output of bias_batch_sum must be a 1D Matrix");
        }

        if Expr::DIMENSIONS == 2 {
            if lhs.rows() != self.lhs.value.columns() {
                panic!("Invalid dimensions for assignment of bias_batch_sum result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// BiasBatchSumExpr is an EtlExpr
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlExpr<T> for BiasBatchSumExpr<T, Expr> {
    const DIMENSIONS: usize = 1;
    const TYPE: EtlType = EtlType::Smart;

    fn size(&self) -> usize {
        self.lhs.value.columns()
    }

    fn rows(&self) -> usize {
        self.lhs.value.columns()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        self.validate_bias_batch_sum(lhs);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_bias_batch_sum(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_bias_batch_sum_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_bias_batch_sum_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_bias_batch_sum_scale(output);
    }

    fn at(&self, i: usize) -> T {
        self.temp[i]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.temp
    }
}

// BiasBatchSumExpr is an EtlWrappable
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlWrappable<T> for BiasBatchSumExpr<T, Expr> {
    type WrappedAs = BiasBatchSumExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// BiasBatchSumExpr computes as copy
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlComputable<T> for BiasBatchSumExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn bias_batch_sum<T: EtlValueType, Expr: WrappableExpr<T>>(lhs: Expr) -> BiasBatchSumExpr<T, Expr> {
    BiasBatchSumExpr::<T, Expr>::new(lhs)
}

crate::impl_add_op_unary_expr!(BiasBatchSumExpr<T, Expr>);
crate::impl_sub_op_unary_expr!(BiasBatchSumExpr<T, Expr>);
crate::impl_mul_op_unary_expr!(BiasBatchSumExpr<T, Expr>);
crate::impl_scale_op_unary_expr!(BiasBatchSumExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::bias_batch_sum_expr::bias_batch_sum;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn bias_batch_sum_simple() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        c |= bias_batch_sum(&a);

        assert_eq!(c.at(0), 9);
        assert_eq!(c.at(1), 12);
    }
}
