use super::etl_expr::*;
use super::matrix_2d::Matrix2d;
use super::vector::Vector;

// TODO: Propagate get_data on smart expression
// TODO: Generalize everywhere
// TODO: Try to get rid of to_vector/to_matrix

fn forward_data_binary<
    T: EtlValueType,
    F: Fn(&mut Vec<T>, &Vec<T>, &Vec<T>),
    LeftExpr: EtlComputable<T> + EtlExpr<T>,
    RightExpr: EtlComputable<T> + EtlExpr<T>,
>(
    output: &mut Vec<T>,
    lhs: &LeftExpr,
    rhs: &RightExpr,
    functor: F,
) {
    if LeftExpr::TYPE == EtlType::Value && RightExpr::TYPE == EtlType::Value {
        functor(output, lhs.get_data(), rhs.get_data());
    } else if LeftExpr::TYPE == EtlType::Value && RightExpr::TYPE != EtlType::Value {
        let rhs_data = rhs.to_data();

        functor(output, lhs.get_data(), &rhs_data);
    } else if LeftExpr::TYPE != EtlType::Value && RightExpr::TYPE == EtlType::Value {
        let lhs_data = lhs.to_data();

        functor(output, &lhs_data, rhs.get_data());
    } else {
        let lhs_data = lhs.to_data();
        let rhs_data = rhs.to_data();

        functor(output, &lhs_data, &rhs_data);
    }
}

// The declaration of MulExpr

/// Expression represneting a vector-matrix-multiplication
/// LeftExpr is a vector expression
/// RightExpr is a matrix expression
/// MulExpr is a vector expression
pub struct MulExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> {
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of MulExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> MulExpr<T, LeftExpr, RightExpr> {
    pub fn new(lhs: LeftExpr, rhs: RightExpr) -> Self {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != rhs.rows() {
                panic!("Invalid vector matrix multiplication dimensions ([{}]*[{},{}])", lhs.rows(), rhs.rows(), rhs.columns());
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            if lhs.columns() != rhs.rows() {
                panic!("Invalid matrix vector multiplication dimensions ([{},{}]*[{}])", lhs.rows(), rhs.rows(), rhs.columns());
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            if lhs.columns() != rhs.rows() {
                panic!(
                    "Invalid matrix matrix multiplication dimensions ([{},{}]*[{},{}])",
                    lhs.rows(),
                    lhs.columns(),
                    rhs.rows(),
                    rhs.columns()
                );
            }
        } else {
            panic!("Invalid vector matrix multiplication dimensions ({}D*{}D)", LeftExpr::DIMENSIONS, RightExpr::DIMENSIONS);
        }

        Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
            temp: Vec::<T>::new(),
        }
    }

    fn compute_gemm(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            output[..self.temp.len()].copy_from_slice(&self.temp[..]);
            return;
        }

        self.compute_gemm_impl(output);
    }

    fn compute_gemm_add(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] += self.temp[n];
            }
            return;
        }

        self.compute_gemm_impl(output);
    }

    fn compute_gemm_sub(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] -= self.temp[n];
            }
            return;
        }

        self.compute_gemm_impl(output);
    }

    fn compute_gemm_scale(&self, output: &mut Vec<T>) {
        // If we already computed the value at construction, we can do a simple copy
        if !self.temp.is_empty() {
            for n in 0..self.temp.len() {
                output[n] *= self.temp[n];
            }
            return;
        }

        self.compute_gemm_impl(output);
    }

    fn compute_gemm_impl(&self, output: &mut Vec<T>) {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            // No need to zero the vector since we did that a construction

            let m = self.rhs.value.rows();
            let n = self.rhs.value.columns();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                for row in 0..m {
                    for column in 0..n {
                        out[column] += lhs[row] * rhs[row * n + column];
                    }
                }
            };

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, functor);
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            // No need to zero the vector since we did that a construction

            let m = self.lhs.value.rows();
            let n = self.lhs.value.columns();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                for row in 0..m {
                    for column in 0..n {
                        out[row] += rhs[column] * lhs[row * n + column];
                    }
                }
            };

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, functor);
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            // No need to zero the vector since we did that a construction

            let m = self.lhs.value.rows();
            let n = self.lhs.value.columns();
            let k = self.rhs.value.columns();

            let functor = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                for row in 0..m {
                    for column in 0..n {
                        for outer in 0..k {
                            out[row * k + outer] += lhs[row * n + column] * rhs[column * k + outer]
                        }
                    }
                }
            };

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, functor);
        } else {
            panic!("This code should be unreachable!");
        }
    }

    fn validate_gemm<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != self.rhs.value.columns() {
                panic!("Invalid dimensions for assignment of GEVM result");
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            if lhs.rows() != self.lhs.value.rows() {
                panic!("Invalid dimensions for assignment of GEMV result");
            }
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            if lhs.rows() != self.lhs.value.rows() || lhs.columns() != self.rhs.value.columns() {
                panic!("Invalid dimensions for assignment of GEMM result");
            }
        } else {
            panic!("This code should be unreachable!");
        }
    }
}

// MulExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for MulExpr<T, LeftExpr, RightExpr> {
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 { 2 } else { 1 };
    const TYPE: EtlType = EtlType::Smart;

    fn size(&self) -> usize {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            self.rhs.value.columns()
        } else if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 1 {
            self.lhs.value.rows()
        } else {
            self.lhs.value.rows() * self.rhs.value.columns()
        }
    }

    fn rows(&self) -> usize {
        if LeftExpr::DIMENSIONS == 1 && RightExpr::DIMENSIONS == 2 {
            self.rhs.value.columns()
        } else {
            self.lhs.value.rows()
        }
    }

    fn columns(&self) -> usize {
        self.rhs.value.columns()
    }

    fn validate_assign<OutputExpr: EtlExpr<T>>(&self, lhs: &OutputExpr) {
        self.validate_gemm(lhs);
    }

    fn compute_into(&self, output: &mut Vec<T>) {
        self.compute_gemm(output);
    }

    fn compute_into_add(&self, output: &mut Vec<T>) {
        self.compute_gemm_add(output);
    }

    fn compute_into_sub(&self, output: &mut Vec<T>) {
        self.compute_gemm_sub(output);
    }

    fn compute_into_scale(&self, output: &mut Vec<T>) {
        self.compute_gemm_scale(output);
    }

    fn at(&self, i: usize) -> T {
        self.temp[i]
    }

    fn at2(&self, row: usize, column: usize) -> T {
        self.temp[row * self.columns() + column]
    }
}

// MulExpr is an EtlWrappable
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for MulExpr<T, LeftExpr, RightExpr> {
    type WrappedAs = MulExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// MulExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for MulExpr<T, LeftExpr, RightExpr> {
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

    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

// Unfortunately, because of the Orphan rule, we cannot implement this trait for each structure
// implementing EtlExpr
// Therefore, we provide macros for other structures and expressions

// Unfortunately, "associated const equality" is an incomplete feature in Rust (issue 92827)
// Therefore, we need to use the same struct for each multiplication and then use if statements to
// detect the actual operation (gemm, gemv, gemv)

#[macro_export]
macro_rules! impl_mul_op_value {
    ($type:ty) => {
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Mul<RightExpr> for &'a $type {
            type Output = $crate::etl::mul_expr::MulExpr<T, &'a $type, RightExpr>;

            fn mul(self, other: RightExpr) -> Self::Output {
                let mut expr = Self::Output::new(self, other);

                if Self::Output::DIMENSIONS == 2 {
                    let temp = expr.to_matrix();
                    expr.temp = temp.value.data;
                } else {
                    let temp = expr.to_vector();
                    expr.temp = temp.value.data;
                }

                expr
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr>
            for $type
        {
            type Output = $crate::etl::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                let mut expr = Self::Output::new(self, other);

                if Self::Output::DIMENSIONS == 2 {
                    let temp = expr.to_matrix();
                    expr.temp = temp.value.data;
                } else {
                    let temp = expr.to_vector();
                    expr.temp = temp.value.data;
                }

                expr
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type {
            type Output = $crate::etl::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                let mut expr = Self::Output::new(self, other);
                expr.temp = expr.to_data();
                expr
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_unary_expr_float {
    ($type:ty) => {
        impl<T: EtlValueType + Float, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type {
            type Output = $crate::etl::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                let mut expr = Self::Output::new(self, other);
                expr.temp = expr.to_data();
                expr
            }
        }
    };
}

crate::impl_add_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr!(MulExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::etl::constant::cst;
    use crate::etl::etl_expr::EtlExpr;
    use crate::etl::matrix_2d::Matrix2d;
    use crate::etl::vector::Vector;

    #[test]
    fn gevm() {
        let mut a = Vector::<i64>::new(3);
        let mut b = Matrix2d::<i64>::new(3, 2);
        let mut c = Vector::<i64>::new(2);

        a[0] = 7;
        a[1] = 8;
        a[2] = 9;

        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
        b[3] = 4;
        b[4] = 5;
        b[5] = 6;

        c |= &a * &b;

        assert_eq!(c.at(0), 76);
        assert_eq!(c.at(1), 100);
    }

    #[test]
    fn basic_gevm_one() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        let expr = &a * &b;
        assert_eq!(expr.size(), 8);
        assert_eq!(expr.rows(), 8);

        c |= expr;
        assert_eq!(c.at(0), 24);
    }

    #[test]
    fn basic_gevm_one_plus() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at(0), 48);
    }

    #[test]
    fn basic_gevm_one_plus_plus() {
        let mut a = Vector::<i64>::new(4);
        let mut b = Matrix2d::<i64>::new(4, 8);
        let mut c = Vector::<i64>::new(8);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at(0), 96);
    }

    #[test]
    fn gemv() {
        let mut a = Matrix2d::<i64>::new(2, 3);
        let mut b = Vector::<i64>::new(3);
        let mut c = Vector::<i64>::new(2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;
        b[2] = 9;

        c |= &a * &b;

        assert_eq!(c.at(0), 50);
        assert_eq!(c.at(1), 122);
    }

    #[test]
    fn basic_gemv_one() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        let expr = &a * &b;
        assert_eq!(expr.size(), 4);
        assert_eq!(expr.rows(), 4);

        c |= expr;
        assert_eq!(c.at(0), 48);
    }

    #[test]
    fn basic_gemv_one_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at(0), 96);
    }

    #[test]
    fn basic_gemv_one_plus_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Vector::<i64>::new(8);
        let mut c = Vector::<i64>::new(4);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at(0), 192);
    }

    #[test]
    fn gemm_a() {
        let mut a = Matrix2d::<i64>::new(2, 3);
        let mut b = Matrix2d::<i64>::new(3, 2);
        let mut c = Matrix2d::<i64>::new(2, 2);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;
        b[2] = 9;
        b[3] = 10;
        b[4] = 11;
        b[5] = 12;

        c |= &a * &b;

        assert_eq!(c.at2(0, 0), 58);
        assert_eq!(c.at2(0, 1), 64);
        assert_eq!(c.at2(1, 0), 139);
        assert_eq!(c.at2(1, 1), 154);

        c |= cst(1) + (&a * &b);

        assert_eq!(c.at2(0, 0), 59);
        assert_eq!(c.at2(0, 1), 65);
        assert_eq!(c.at2(1, 0), 140);
        assert_eq!(c.at2(1, 1), 155);

        c |= (&a * &b) - cst(1);

        assert_eq!(c.at2(0, 0), 57);
        assert_eq!(c.at2(0, 1), 63);
        assert_eq!(c.at2(1, 0), 138);
        assert_eq!(c.at2(1, 1), 153);
    }

    #[test]
    fn gemm_a_compound_add() {
        let mut a = Matrix2d::<i64>::new(2, 3);
        let mut b = Matrix2d::<i64>::new(3, 2);
        let mut c = Matrix2d::<i64>::new(2, 2);

        c.fill(10);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;
        b[2] = 9;
        b[3] = 10;
        b[4] = 11;
        b[5] = 12;

        c += &a * &b;

        assert_eq!(c.at2(0, 0), 68);
        assert_eq!(c.at2(0, 1), 74);
        assert_eq!(c.at2(1, 0), 149);
        assert_eq!(c.at2(1, 1), 164);
    }

    #[test]
    fn gemm_a_compound_sub() {
        let mut a = Matrix2d::<i64>::new(2, 3);
        let mut b = Matrix2d::<i64>::new(3, 2);
        let mut c = Matrix2d::<i64>::new(2, 2);

        c.fill(10);

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;
        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;
        b[2] = 9;
        b[3] = 10;
        b[4] = 11;
        b[5] = 12;

        c -= &a * &b;

        assert_eq!(c.at2(0, 0), 10 - 58);
        assert_eq!(c.at2(0, 1), 10 - 64);
        assert_eq!(c.at2(1, 0), 10 - 139);
        assert_eq!(c.at2(1, 1), 10 - 154);
    }

    #[test]
    fn gemm_b() {
        let mut a = Matrix2d::<i64>::new(3, 3);
        let mut b = Matrix2d::<i64>::new(3, 3);
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

        b[0] = 7;
        b[1] = 8;
        b[2] = 9;
        b[3] = 9;
        b[4] = 10;
        b[5] = 11;
        b[6] = 11;
        b[7] = 12;
        b[8] = 13;

        c |= &a * &b;

        assert_eq!(c.at2(0, 0), 58);
        assert_eq!(c.at2(0, 1), 64);
        assert_eq!(c.at2(0, 2), 70);
        assert_eq!(c.at2(1, 0), 139);
        assert_eq!(c.at2(1, 1), 154);
        assert_eq!(c.at2(1, 2), 169);
        assert_eq!(c.at2(2, 0), 220);
        assert_eq!(c.at2(2, 1), 244);
        assert_eq!(c.at2(2, 2), 268);
    }

    #[test]
    fn basic_gemm_one() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        let expr = &a * &b;
        assert_eq!(expr.size(), 24);
        assert_eq!(expr.rows(), 4);
        assert_eq!(expr.columns(), 6);

        c |= expr;
        assert_eq!(c.at2(0, 0), 48);
    }

    #[test]
    fn basic_gemm_one_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        c |= (&a * &b) + (&a * &b);

        assert_eq!(c.at2(0, 0), 96);
    }

    #[test]
    fn basic_gemm_one_plus_plus() {
        let mut a = Matrix2d::<i64>::new(4, 8);
        let mut b = Matrix2d::<i64>::new(8, 6);
        let mut c = Matrix2d::<i64>::new(4, 6);

        a.fill(2);
        b.fill(3);

        c |= (&a + &a) * (&b + &b);

        assert_eq!(c.at2(0, 0), 192);
    }

    #[test]
    fn basic_gemm_chain() {
        let mut a = Matrix2d::<i64>::new(3, 3);
        let mut b = Matrix2d::<i64>::new(3, 3);
        let mut c = Matrix2d::<i64>::new(3, 3);
        let mut d = Matrix2d::<i64>::new(3, 3);

        a.fill(2);
        b.fill(3);
        c.fill(4);

        d |= &a * &b * &c;

        assert_eq!(d.at2(0, 0), 216);
    }
}
