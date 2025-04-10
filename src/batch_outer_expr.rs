use crate::etl_expr::*;

// The declaration of BiasOuterExpr

/// Expression representing the batched addition of biases to a matrix
pub struct BiasOuterExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of BiasOuterExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> BiasOuterExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != rhs.rows() {
                panic!("Invalid batch_outer dimensions ([{},{}]*[{},{}])", lhs.rows(), lhs.columns(), rhs.rows(), rhs.columns());
            }
        } else {
            panic!("Invalid batch_outer dimensions ({}D*{}D)", LeftExpr::DIMENSIONS, RightExpr::DIMENSIONS);
        }

        let mut expr = Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_batch_outer_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_batch_outer(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            output[..self.temp.len()].copy_from_slice(&self.temp[..]);
            return;
        }

        self.compute_batch_outer_impl(output);
    }

    fn compute_batch_outer_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] += self.temp[n];
            }
            return;
        }

        self.compute_batch_outer_impl(output);
    }

    fn compute_batch_outer_sub(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] -= self.temp[n];
            }
            return;
        }

        self.compute_batch_outer_impl(output);
    }

    fn compute_batch_outer_scale(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] *= self.temp[n];
            }
            return;
        }

        self.compute_batch_outer_impl(output);
    }

    fn compute_batch_outer_impl(&self, output: &mut Vec<T>) {
        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            let m = self.lhs.value.columns();
            let n = self.rhs.value.columns();
            let b = self.lhs.value.rows();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                for batch in 0..b {
                    for row in 0..m {
                        for column in 0..n {
                            out[row * n + column] += lhs[batch * m + row] + rhs[batch * n + column];
                        }
                    }
                }
            };

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_batch_outer<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        if OutputExpr::DIMENSIONS != 2 {
            panic!("The output of batch_outer must be a 2D Matrix");
        }

        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != self.lhs.value.columns() || lhs.columns() != self.rhs.value.columns() {
                panic!("Invalid dimensions for assignment of batch_outer result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// BiasOuterExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for BiasOuterExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Smart;

    fn size(&self) -> usize {
        self.lhs.value.columns() * self.rhs.value.columns()
    }

    fn rows(&self) -> usize {
        self.lhs.value.columns()
    }

    fn columns(&self) -> usize {
        self.rhs.value.columns()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        self.validate_batch_outer(lhs);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_batch_outer(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_batch_outer_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_batch_outer_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_batch_outer_scale(output);
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

// BiasOuterExpr is an EtlWrappable
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for BiasOuterExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = BiasOuterExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// BiasOuterExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for BiasOuterExpr<T, LeftExpr, RightExpr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn batch_outer<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>(
    lhs: LeftExpr,
    rhs: RightExpr,
) -> BiasOuterExpr<T, LeftExpr, RightExpr> {
    BiasOuterExpr::<T, LeftExpr, RightExpr>::new(lhs, rhs)
}

crate::impl_add_op_binary_expr!(BiasOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(BiasOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(BiasOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(BiasOuterExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::batch_outer_expr::batch_outer;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;

    #[test]
    fn batch_outer_simple() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut b = Matrix2d::<i64>::new(3, 4);
        let mut c = Matrix2d::<i64>::new(2, 4);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 11;
        b[1] = 12;
        b[2] = 13;
        b[3] = 14;
        b[4] = 15;
        b[5] = 16;
        b[6] = 17;
        b[7] = 18;
        b[8] = 19;
        b[9] = 20;
        b[10] = 21;
        b[11] = 22;

        c |= batch_outer(&a, &b);

        assert_eq!(c.at2(0, 0), 54);
        assert_eq!(c.at2(0, 1), 57);
        assert_eq!(c.at2(0, 2), 60);
        assert_eq!(c.at2(0, 3), 63);
        assert_eq!(c.at2(1, 0), 57);
        assert_eq!(c.at2(1, 1), 60);
        assert_eq!(c.at2(1, 2), 63);
        assert_eq!(c.at2(1, 3), 66);
    }

    #[test]
    fn batch_outer_deep() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut b = Matrix2d::<i64>::new(3, 4);
        let mut c = Matrix2d::<i64>::new(2, 4);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 11;
        b[1] = 12;
        b[2] = 13;
        b[3] = 14;
        b[4] = 15;
        b[5] = 16;
        b[6] = 17;
        b[7] = 18;
        b[8] = 19;
        b[9] = 20;
        b[10] = 21;
        b[11] = 22;

        c |= batch_outer(&a, &b) + batch_outer(&a, &b);

        assert_eq!(c.at2(0, 0), 108);
        assert_eq!(c.at2(0, 1), 114);
        assert_eq!(c.at2(0, 2), 120);
        assert_eq!(c.at2(0, 3), 126);
        assert_eq!(c.at2(1, 0), 114);
        assert_eq!(c.at2(1, 1), 120);
        assert_eq!(c.at2(1, 2), 126);
        assert_eq!(c.at2(1, 3), 132);
    }
}
