use crate::etl::add_expr::AddExpr;
use crate::etl::etl_expr::*;
use crate::etl::mul_expr::MulExpr;

use crate::impl_add_op_binary_expr;
use crate::impl_mul_op_binary_expr;

use crate::etl::matrix_2d::Matrix2d;
use crate::etl::vector::Vector;

// The declaration of SigmoidExpr

pub struct SigmoidExpr<T: EtlValueType + Float, Expr: WrappableExpr<T>> {
    expr: EtlWrapper<T, Expr::WrappedAs>,
}

// The functions of SigmoidExpr

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> SigmoidExpr<T, Expr> {
    pub fn new(expr: Expr) -> Self {
        Self { expr: expr.wrap() }
    }
}

// SigmoidExpr is an EtlExpr
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for SigmoidExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = EtlType::Simple;

    fn size(&self) -> usize {
        self.expr.value.size()
    }

    fn rows(&self) -> usize {
        self.expr.value.rows()
    }

    fn columns(&self) -> usize {
        self.expr.value.columns()
    }

    fn at(&self, i: usize) -> T {
        T::one() / (T::one() + self.expr.value.at(i).exp())
    }

    fn at2(&self, row: usize, column: usize) -> T {
        T::one() / (T::one() + self.expr.value.at2(row, column).exp())
    }
}

impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlExpr<T> for &SigmoidExpr<T, Expr> {
    const DIMENSIONS: usize = Expr::DIMENSIONS;
    const TYPE: EtlType = EtlType::Simple;

    fn size(&self) -> usize {
        self.expr.value.size()
    }

    fn rows(&self) -> usize {
        self.expr.value.rows()
    }

    fn columns(&self) -> usize {
        self.expr.value.columns()
    }

    fn at(&self, i: usize) -> T {
        T::one() / (T::one() + self.expr.value.at(i).exp())
    }

    fn at2(&self, row: usize, column: usize) -> T {
        T::one() / (T::one() + self.expr.value.at2(row, column).exp())
    }
}

// SigmoidExpr is an EtlWrappable
// SigmoidExpr wraps as value
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlWrappable<T> for SigmoidExpr<T, Expr> {
    type WrappedAs = SigmoidExpr<T, Expr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// SigmoidExpr computes as copy
impl<T: EtlValueType + Float, Expr: WrappableExpr<T>> EtlComputable<T> for SigmoidExpr<T, Expr> {
    type ComputedAsVector = Vector<T>;
    type ComputedAsMatrix = Matrix2d<T>;

    fn to_vector(&self) -> EtlWrapper<T, Self::ComputedAsVector> {
        let mut vec = Vector::<T>::new(self.rows());
        assign_direct(&mut vec.data, self);
        EtlWrapper {
            value: vec,
            _marker: std::marker::PhantomData,
        }
    }

    fn to_matrix(&self) -> EtlWrapper<T, Self::ComputedAsMatrix> {
        let mut vec = Matrix2d::<T>::new(self.rows(), self.columns());
        assign_direct(&mut vec.data, self);
        EtlWrapper {
            value: vec,
            _marker: std::marker::PhantomData,
        }
    }
}

// Operations

pub fn sigmoid<T: EtlValueType + Float, Expr: WrappableExpr<T>>(expr: Expr) -> SigmoidExpr<T, Expr> {
    SigmoidExpr::<T, Expr>::new(expr)
}

// TODO Allow chaining operators
//impl_add_op_binary_expr!(SigmoidExpr<T, Expr, RightExpr>);
//impl_sub_op_binary_expr!(SigmoidExpr<T, Expr, RightExpr>);
//impl_mul_op_binary_expr!(SigmoidExpr<T, Expr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::sigmoid_expr::sigmoid;
    use crate::etl::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_sigmoid() {
        let mut a = Vector::<f64>::new(5);
        let mut b = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;

        let expr = sigmoid(&a);

        assert_eq!(expr.size(), 5);
        assert_relative_eq!(expr.at(0), 0.268941421, epsilon = 1e-6);

        b |= sigmoid(&a);

        assert_relative_eq!(b.at(0), 0.268941421, epsilon = 1e-6);
        assert_relative_eq!(b.at(1), 0.119202922, epsilon = 1e-6);
        assert_relative_eq!(b.at(2), 0.047425872, epsilon = 1e-6);
        assert_relative_eq!(b.at(3), 0.017986209, epsilon = 1e-6);
        assert_relative_eq!(b.at(4), 0.006692850, epsilon = 1e-6);
    }
}
