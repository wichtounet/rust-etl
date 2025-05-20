use crate::etl_expr::*;

// The declaration of ArgMaxExpr

pub struct ArgMaxExpr<T: EtlValueType, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of ArgMaxExpr

impl<T: EtlValueType, Expr: WrappableExpr<T>> ArgMaxExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        if Expr::DIMENSIONS != 2 {
            panic!("argmax only works on 2D expression");
        }

        let mut expr = Self {
            expr: expr.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_argmax_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_argmax(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        output[..self.temp.len()].copy_from_slice(&self.temp[..]);
    }

    fn compute_argmax_add(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs += *rhs;
        }
    }

    fn compute_argmax_sub(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs -= *rhs;
        }
    }

    fn compute_argmax_scale(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs *= *rhs;
        }
    }

    fn compute_argmax_div(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (lhs, rhs) in output.iter_mut().zip(self.temp.iter()) {
            *lhs /= *rhs;
        }
    }

    fn compute_argmax_impl(&self, output: &mut Vec<T>) {
        if Expr::DIMENSIONS == 2 {
            let rows = self.expr.value.rows();
            let columns = self.expr.value.columns();

            let functor = |out: &mut [T], expr: &[T]| {
                for row in 0..rows {
                    let mut max_index = 0;
                    let mut current_return_index = T::zero();
                    let mut return_index = T::zero();

                    for column in 1..columns {
                        current_return_index += T::one();

                        if expr[row * columns + column] > expr[row * columns + max_index] {
                            max_index = column;
                            return_index = current_return_index;
                        }
                    }

                    out[row] = return_index
                }
            };

            forward_data_unary(output, &self.expr.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_batch_softmax<OutputExpr: EtlExpr<T>>(&self, expr: &OutputExpr) {
        if OutputExpr::DIMENSIONS != 1 {
            panic!("The output of argmax must be a 1D Vector");
        }

        if Expr::DIMENSIONS == 2 {
            if expr.size() != self.expr.value.rows() {
                panic!("Invalid dimensions for assignment of batch_softmax result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// ArgMaxExpr is an EtlExpr
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlExpr<T> for ArgMaxExpr<T, Expr> {
    const DIMENSIONS: usize = 1;
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
        self.compute_argmax(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_argmax_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_argmax_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_argmax_scale(output);
    }

    fn compute_into_div(&self, output: &mut Vec<T>) {
        self.compute_argmax_div(output);
    }

    fn at(&self, i: usize) -> T {
        self.temp[i]
    }

    fn get_data(&self) -> &[T] {
        &self.temp
    }
}

// ArgMaxExpr is an EtlWrappable
// ArgMaxExpr wraps as value
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlWrappable<T> for ArgMaxExpr<T, Expr> {
    type WrappedAs = ArgMaxExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// ArgMaxExpr computes as copy
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlComputable<T> for ArgMaxExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn argmax<T: EtlValueType, Expr: WrappableExpr<T>>(expr: Expr) -> ArgMaxExpr<T, Expr> {
    ArgMaxExpr::<T, Expr>::new(expr)
}

crate::impl_add_op_unary_expr!(ArgMaxExpr<T, Expr>);
crate::impl_sub_op_unary_expr!(ArgMaxExpr<T, Expr>);
crate::impl_mul_op_unary_expr!(ArgMaxExpr<T, Expr>);
crate::impl_div_op_unary_expr!(ArgMaxExpr<T, Expr>);
crate::impl_scale_op_unary_expr!(ArgMaxExpr<T, Expr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::argmax_expr::argmax;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

    #[test]
    fn basic_argmax() {
        let mut a = Matrix2d::<i64>::new(2, 3);
        let mut b = Vector::<i64>::new(2);

        *a.at_mut(0, 0) = 1;
        *a.at_mut(0, 1) = 2;
        *a.at_mut(0, 2) = 3;

        *a.at_mut(1, 0) = 5;
        *a.at_mut(1, 1) = 2;
        *a.at_mut(1, 2) = 3;

        b |= argmax(&a);

        assert_eq!(b.at(0), 2);
        assert_eq!(b.at(1), 0);
    }
}
