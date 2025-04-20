use crate::etl_expr::*;

// The declaration of TransposeExpr

pub struct TransposeExpr<T: EtlValueType, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of TransposeExpr

impl<T: EtlValueType, Expr: WrappableExpr<T>> TransposeExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        if Expr::DIMENSIONS != 2 {
            panic!("Invalid vector matrix multiplication dimensions ({}D)", Expr::DIMENSIONS);
        }

        let mut expr = Self {
            expr: expr.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_transpose_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_transpose(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            output[..self.temp.len()].copy_from_slice(&self.temp[..]);
            return;
        }

        self.compute_transpose_impl(output);
    }

    fn compute_transpose_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] += self.temp[n];
            }
            return;
        }

        self.compute_transpose_impl(output);
    }

    fn compute_transpose_sub(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] -= self.temp[n];
            }
            return;
        }

        self.compute_transpose_impl(output);
    }

    fn compute_transpose_scale(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] *= self.temp[n];
            }
            return;
        }

        self.compute_transpose_impl(output);
    }

    fn compute_transpose_impl(&self, output: &mut Vec<T>) {
        if Expr::DIMENSIONS == 2 {
            let m = self.expr.value.rows();
            let n = self.expr.value.columns();

            let functor = |out: &mut Vec<T>, expr: &Vec<T>| {
                for row in 0..m {
                    for column in 0..n {
                        out[column * m + row] = expr[row * n + column];
                    }
                }
            };

            forward_data_unary(output, &self.expr.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_transpose<OutputExpr: EtlExpr<T>>(&self, expr: &OutputExpr) {
        if Expr::DIMENSIONS == 2 {
            if expr.rows() != self.expr.value.columns() || expr.columns() != self.expr.value.rows() {
                panic!("Invalid dimensions for assignment of GEMM result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// TransposeExpr is an EtlExpr
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlExpr<T> for TransposeExpr<T, Expr> {
    const DIMENSIONS: usize = 2;
    const TYPE: EtlType = EtlType::Smart;

    fn size(&self) -> usize {
        self.expr.value.size()
    }

    fn rows(&self) -> usize {
        self.expr.value.columns()
    }

    fn columns(&self) -> usize {
        self.expr.value.rows()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, expr: &OutputExpr) {
        self.validate_transpose(expr);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_transpose(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_transpose_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_transpose_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_transpose_scale(output);
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

// TransposeExpr is an EtlWrappable
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlWrappable<T> for TransposeExpr<T, Expr> {
    type WrappedAs = TransposeExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// TransposeExpr computes as copy
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlComputable<T> for TransposeExpr<T, Expr> {
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

crate::impl_add_op_unary_expr!(TransposeExpr<T, Expr>);
crate::impl_sub_op_unary_expr!(TransposeExpr<T, Expr>);
crate::impl_mul_op_unary_expr!(TransposeExpr<T, Expr>);
crate::impl_div_op_unary_expr!(TransposeExpr<T, Expr>);
crate::impl_scale_op_unary_expr!(TransposeExpr<T, Expr>);

pub fn transpose<T: EtlValueType, Expr: WrappableExpr<T>>(expr: Expr) -> TransposeExpr<T, Expr> {
    TransposeExpr::<T, Expr>::new(expr)
}

// The tests

#[cfg(test)]
mod tests {
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::transpose_expr::transpose;

    #[test]
    fn transpose_a() {
        let mut a = Matrix2d::<i64>::new(3, 2);
        let mut c = Matrix2d::<i64>::new(2, 3);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        c |= transpose(&a);

        assert_eq!(c.at2(0, 0), a.at2(0, 0));
        assert_eq!(c.at2(1, 0), a.at2(0, 1));
        assert_eq!(c.at2(0, 1), a.at2(1, 0));
        assert_eq!(c.at2(1, 1), a.at2(1, 1));
        assert_eq!(c.at2(0, 2), a.at2(2, 0));
        assert_eq!(c.at2(1, 2), a.at2(2, 1));
    }

    #[test]
    fn transpose_b() {
        let mut a = Matrix2d::<i64>::new(3, 3);
        let mut c = Matrix2d::<i64>::new(3, 3);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;
        a[6] = 7;
        a[7] = 8;
        a[8] = 9;

        c |= transpose(&a);

        assert_eq!(c.at2(0, 0), a.at2(0, 0));
        assert_eq!(c.at2(1, 0), a.at2(0, 1));
        assert_eq!(c.at2(2, 0), a.at2(0, 2));
        assert_eq!(c.at2(0, 1), a.at2(1, 0));
        assert_eq!(c.at2(1, 1), a.at2(1, 1));
        assert_eq!(c.at2(2, 1), a.at2(1, 2));
        assert_eq!(c.at2(0, 2), a.at2(2, 0));
        assert_eq!(c.at2(1, 2), a.at2(2, 1));
        assert_eq!(c.at2(2, 2), a.at2(2, 2));
    }
}
