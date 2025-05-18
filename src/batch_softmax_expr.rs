use crate::base_traits::Float;
use crate::etl_expr::*;

// The declaration of BatchSoftmaxExpr

/// Expression representing the batched addition of biases to a matrix
pub struct BatchSoftmaxExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of BatchSoftmaxExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> BatchSoftmaxExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        if Expr::DIMENSIONS != 2 {
            panic!("Invalid batch_softmax dimensions ({}D)", Expr::DIMENSIONS);
        }

        let mut expr = Self {
            expr: expr.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_batch_softmax_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_batch_softmax(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        output[..self.temp.len()].copy_from_slice(&self.temp[..]);
    }

    fn compute_batch_softmax_add(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs += *rhs;
        }
    }

    fn compute_batch_softmax_sub(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs -= *rhs;
        }
    }

    fn compute_batch_softmax_scale(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs *= *rhs;
        }
    }

    fn compute_batch_softmax_div(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs /= *rhs;
        }
    }

    fn compute_batch_softmax_impl(&self, output: &mut Vec<T>) {
        if Expr::DIMENSIONS == 2 {
            let b = self.expr.value.rows();
            let m = self.expr.value.columns();

            let functor = |out: &mut Vec<T>, expr: &Vec<T>| {
                for batch in 0..b {
                    let mut sum_exp = T::zero();

                    for row in 0..m {
                        sum_exp += (expr[batch * m + row]).exp();
                    }

                    for row in 0..m {
                        out[batch * m + row] = expr[batch * m + row].exp() / sum_exp;
                    }
                }
            };

            forward_data_unary(output, &self.expr.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_batch_softmax<OutputExpr: EtlExpr<T>>(&self, expr: &OutputExpr) {
        if OutputExpr::DIMENSIONS != 2 {
            panic!("The output of batch_softmax must be a 2D Matrix");
        }

        if Expr::DIMENSIONS == 2 {
            if expr.size() != self.expr.value.size() {
                panic!("Invalid dimensions for assignment of batch_softmax result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// BatchSoftmaxExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for BatchSoftmaxExpr<T, Expr> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Smart;
    const THREAD_SAFE: bool = true;

    type Iter<'x>
        = std::iter::Cloned<std::slice::Iter<'x, T>>
    where
        T: 'x,
        Self: 'x;

    fn iter(&self) -> Self::Iter<'_> {
        self.temp.iter().cloned()
    }

    fn iter_range(&self, range: std::ops::Range<usize>) -> Self::Iter<'_> {
        self.temp[range].iter().cloned()
    }

    fn size(&self) -> usize {
        self.expr.value.size()
    }

    fn rows(&self) -> usize {
        self.expr.value.rows()
    }

    fn columns(&self) -> usize {
        self.expr.value.columns()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, expr: &OutputExpr) {
        self.validate_batch_softmax(expr);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_batch_softmax(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_batch_softmax_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_batch_softmax_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_batch_softmax_scale(output);
    }

    fn compute_into_div(&self, output: &mut Vec<T>) {
        self.compute_batch_softmax_div(output);
    }

    fn at(&self, i: usize) -> T {
        self.temp[i]
    }

    fn get_data(&self) -> &Vec<T> {
        &self.temp
    }
}

// BatchSoftmaxExpr is an EtlWrappable
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for BatchSoftmaxExpr<T, Expr> {
    type WrappedAs = BatchSoftmaxExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// BatchSoftmaxExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for BatchSoftmaxExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn batch_softmax<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> BatchSoftmaxExpr<T, Expr> {
    BatchSoftmaxExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr_trait!(Float, BatchSoftmaxExpr<T, Expr>);
crate::impl_sub_op_unary_expr_trait!(Float, BatchSoftmaxExpr<T, Expr>);
crate::impl_mul_op_unary_expr_trait!(Float, BatchSoftmaxExpr<T, Expr>);
crate::impl_div_op_unary_expr_trait!(Float, BatchSoftmaxExpr<T, Expr>);
crate::impl_scale_op_unary_expr_trait!(Float, BatchSoftmaxExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::batch_softmax_expr::batch_softmax;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use approx::assert_relative_eq;

    #[test]
    fn batch_softmax_simple() {
        let mut a = Matrix2d::<f64>::new(3, 2);
        let mut c = Matrix2d::<f64>::new(3, 2);

        a[0] = 1.0;
        a[1] = 2.0;

        a[2] = 3.0;
        a[3] = 4.0;

        a[4] = 5.0;
        a[5] = 9.0;

        c |= batch_softmax(&a);

        assert_relative_eq!(c.at2(0, 0), 0.268941, epsilon = 1e-6);
        assert_relative_eq!(c.at2(0, 1), 0.731058, epsilon = 1e-6);
        assert_relative_eq!(c.at2(1, 0), 0.268941, epsilon = 1e-6);
        assert_relative_eq!(c.at2(1, 1), 0.731058, epsilon = 1e-6);
        assert_relative_eq!(c.at2(2, 0), 0.017986, epsilon = 1e-6);
        assert_relative_eq!(c.at2(2, 1), 0.982013, epsilon = 1e-6);
    }
}
