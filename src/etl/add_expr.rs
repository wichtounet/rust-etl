use super::etl_expr::*;
use super::matrix_2d::Matrix2d;
use super::vector::Vector;

// The declaration of AddExpr

pub struct AddExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
}

// The functions of AddExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> AddExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS > 0 && RightExpr::DIMENSIONS > 0 && lhs.size() != rhs.size() {
            panic!("Cannot add expressions of different sizes ({} + {})", lhs.size(), rhs.size());
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
        }
    }
}

// AddExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for AddExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS > 0 {
        LeftExpr::DIMENSIONS
    } else {
        RightExpr::DIMENSIONS
    };
    const TYPE: EtlType = EtlType::Simple;

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.size()
        } else {
            self.rhs.value.size()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.rows()
        } else {
            self.rhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.columns()
        } else {
            self.rhs.value.columns()
        }
    }

    fn at(&self, i: usize) -> T {
        self.lhs.value.at(i) + self.rhs.value.at(i)
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.lhs.value.at2(row, column) + self.rhs.value.at2(row, column)
    }
}

// AddExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for &AddExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = LeftExpr::DIMENSIONS;
    const TYPE: EtlType = EtlType::Simple;

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.size()
        } else {
            self.rhs.value.size()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.rows()
        } else {
            self.rhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        if LeftExpr::DIMENSIONS > 0 {
            self.lhs.value.columns()
        } else {
            self.rhs.value.columns()
        }
    }

    fn at(&self, i: usize) -> T {
        self.lhs.value.at(i) + self.rhs.value.at(i)
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.lhs.value.at2(row, column) + self.rhs.value.at2(row, column)
    }
}

// AddExpr is an EtlWrappable
// AddExpr wraps as value
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for AddExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = AddExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// AddExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for AddExpr<T, LeftExpr, RightExpr> {
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

// Unfortunately, because of the Orphan rule, we cannot implement this trait for each structure
// implementing EtlExpr
// Therefore, we provide macros for other structures and expressions

#[macro_export]
macro_rules! impl_add_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Add<RightExpr> for &'a $type {
            type Output = $crate::etl::add_expr::AddExpr<T, &'a $type, RightExpr>;

            fn add(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }

        impl<T: EtlValueType, RightExpr: EtlExpr<T>> std::ops::AddAssign<RightExpr> for $type {
            fn add_assign(&mut self, other: RightExpr) {
                add_assign_direct(&mut self.data, other);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_constant {
    ($type:ty) => {
        impl<T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Add<RightExpr> for $type {
            type Output = $crate::etl::add_expr::AddExpr<T, $type, RightExpr>;

            fn add(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Add<OuterRightExpr>
            for $type
        {
            type Output = $crate::etl::add_expr::AddExpr<T, $type, OuterRightExpr>;

            fn add(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Add<OuterRightExpr> for $type {
            type Output = $crate::etl::add_expr::AddExpr<T, $type, OuterRightExpr>;

            fn add(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add_op_unary_expr_float {
    ($type:ty) => {
        impl<T: EtlValueType + Float, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Add<OuterRightExpr> for $type {
            type Output = $crate::etl::add_expr::AddExpr<T, $type, OuterRightExpr>;

            fn add(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

crate::impl_add_op_binary_expr!(AddExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(AddExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(AddExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::matrix_2d::Matrix2d;
    use crate::etl::vector::Vector;

    #[test]
    fn basic_one() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a + &b;

        assert_eq!(expr.size(), 8);
        assert_eq!(expr.at(0), 3);
    }

    #[test]
    fn basic_assign_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        let expr = &a + &b;

        c |= expr;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a + &b;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_mixed() {
        let mut a = Matrix2d::<i64>::new(4, 2);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= &a + &b;

        assert_eq!(c.at(0), 3);
    }

    #[test]
    fn basic_assign_deep_1() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a + &b) + &a;

        assert_eq!(c.at(0), 4);
    }

    #[test]
    fn basic_assign_deep_2() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c |= (&a + &b) + (&a + &b);

        assert_eq!(c.at(0), 6);
    }

    #[test]
    fn basic_compound_add() {
        let mut a = Vector::<i64>::new(8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(8);

        a[0] = 1;
        b[0] = 2;

        c += &a + &b;

        assert_eq!(c.at(0), 3);
    }
}
