use crate::etl_expr::*;

// The declaration of ArgMaxExpr

pub struct ArgMaxExpr<T: EtlValueType, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of ArgMaxExpr

impl<T: EtlValueType, Expr: WrappableExpr<T>> ArgMaxExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        if Expr::DIMENSIONS != 2 {
            panic!("argmax only works on 2D expression");
        }

        Self { expr: expr.wrap() }
    }
}

// ArgMaxExpr is an EtlExpr
impl<T: EtlValueType, Expr: WrappableExpr<T>> EtlExpr<T> for ArgMaxExpr<T, Expr> {
    const DIMENSIONS: usize = 1;
    const TYPE: EtlType = EtlType::Unaligned;

    fn size(&self) -> usize {
        self.expr.value.rows()
    }

    fn rows(&self) -> usize {
        self.expr.value.rows()
    }

    fn at(&self, i: usize) -> T {
        let mut max_index = 0;
        let mut current_return_index = T::zero();
        let mut return_index = T::zero();

        for column in 1..self.expr.value.columns() {
            current_return_index += T::one();

            if self.expr.value.at2(i, column) > self.expr.value.at2(i, max_index) {
                max_index = column;
                return_index = current_return_index;
            }
        }

        return_index
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
        let mut vec = vec![T::default(); padded_size(self.size())];
        assign_direct(&mut vec, self);
        vec
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
