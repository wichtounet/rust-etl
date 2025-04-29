use crate::base_traits::*;
use crate::etl_expr::*;

use std::simd::*;

// The declaration of BatchOuterExpr

/// Expression representing the batched addition of biases to a matrix
pub struct BatchOuterExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>
where
    Simd<T, 8>: SimdHelper,
{
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of BatchOuterExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> BatchOuterExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
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
        assert!(!self.temp.is_empty());

        output[..self.temp.len()].copy_from_slice(&self.temp[..]);
    }

    fn compute_batch_outer_add(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] += self.temp[n];
        }
    }

    fn compute_batch_outer_sub(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] -= self.temp[n];
        }
    }

    fn compute_batch_outer_scale(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] *= self.temp[n];
        }
    }

    fn compute_batch_outer_div(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] /= self.temp[n];
        }
    }

    // Note: Cannot use reduce_sum in generic code
    fn sum(input: &[T; 8]) -> T {
        input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7]
    }

    // For small matrices, it is not worth computing the transpose of lhs and rhs
    // Instead, we unroll the outer loop so that we can compute multiple elements together
    fn small_kernel(m: usize, n: usize, b: usize, out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>) {
        for row in 0..m {
            let mut column = 0;

            while column + 3 < n {
                let c1 = column;
                let c2 = column + 1;
                let c3 = column + 2;
                let c4 = column + 3;

                let mut v1 = T::default();
                let mut v2 = T::default();
                let mut v3 = T::default();
                let mut v4 = T::default();

                for batch in 0..b {
                    v1 += lhs[batch * m + row] * rhs[batch * n + c1];
                    v2 += lhs[batch * m + row] * rhs[batch * n + c2];
                    v3 += lhs[batch * m + row] * rhs[batch * n + c3];
                    v4 += lhs[batch * m + row] * rhs[batch * n + c4];
                }

                out[row * n + c1] = v1;
                out[row * n + c2] = v2;
                out[row * n + c3] = v3;
                out[row * n + c4] = v4;

                column += 4;
            }

            while column + 1 < n {
                let c1 = column;
                let c2 = column + 1;

                let mut v1 = out[row * n + c1];
                let mut v2 = out[row * n + c2];

                for batch in 0..b {
                    v1 += lhs[batch * m + row] * rhs[batch * n + c1];
                    v2 += lhs[batch * m + row] * rhs[batch * n + c2];
                }

                out[row * n + c1] = v1;
                out[row * n + c2] = v2;

                column += 2;
            }

            if column < n {
                let mut v = out[row * n + column];

                for batch in 0..b {
                    v += lhs[batch * m + row] * rhs[batch * n + column];
                }

                out[row * n + column] = v;
            }
        }
    }

    // For medium-to-large matrices, we can transpose lhs and rhs and then we can vectorize the
    // inner loop properly
    fn transposed_kernel(m_start: usize, m_end: usize, n: usize, b: usize, out: &mut [T], lhs_opp: &Vec<T>, rhs_opp: &Vec<T>) {
        let lanes = 8;

        let mut row = m_start;

        while row + 1 < m_end {
            let r1 = row;
            let r2 = row + 1;

            let mut column = 0;

            while column + 3 < n {
                let c1 = column;
                let c2 = column + 1;
                let c3 = column + 2;
                let c4 = column + 3;

                let mut xmm1 = Simd::<T, 8>::splat(T::default());
                let mut xmm2 = Simd::<T, 8>::splat(T::default());
                let mut xmm3 = Simd::<T, 8>::splat(T::default());
                let mut xmm4 = Simd::<T, 8>::splat(T::default());
                let mut xmm5 = Simd::<T, 8>::splat(T::default());
                let mut xmm6 = Simd::<T, 8>::splat(T::default());
                let mut xmm7 = Simd::<T, 8>::splat(T::default());
                let mut xmm8 = Simd::<T, 8>::splat(T::default());

                let mut batch = 0;

                while batch + lanes - 1 < b {
                    let l1 = Simd::<T, 8>::from_slice(&lhs_opp[r1 * b + batch..]);
                    let l2 = Simd::<T, 8>::from_slice(&lhs_opp[r2 * b + batch..]);

                    let r1 = Simd::<T, 8>::from_slice(&rhs_opp[c1 * b + batch..]);
                    let r2 = Simd::<T, 8>::from_slice(&rhs_opp[c2 * b + batch..]);
                    let r3 = Simd::<T, 8>::from_slice(&rhs_opp[c3 * b + batch..]);
                    let r4 = Simd::<T, 8>::from_slice(&rhs_opp[c4 * b + batch..]);

                    // TODO: Missed opportunity to use FMA here
                    xmm1 += l1 * r1;
                    xmm2 += l1 * r2;
                    xmm3 += l1 * r3;
                    xmm4 += l1 * r4;

                    xmm5 += l2 * r1;
                    xmm6 += l2 * r2;
                    xmm7 += l2 * r3;
                    xmm8 += l2 * r4;

                    batch += lanes;
                }

                let mut v11 = Self::sum(&xmm1.to_array());
                let mut v12 = Self::sum(&xmm2.to_array());
                let mut v13 = Self::sum(&xmm3.to_array());
                let mut v14 = Self::sum(&xmm4.to_array());

                let mut v21 = Self::sum(&xmm5.to_array());
                let mut v22 = Self::sum(&xmm6.to_array());
                let mut v23 = Self::sum(&xmm7.to_array());
                let mut v24 = Self::sum(&xmm8.to_array());

                while batch < b {
                    v11 += lhs_opp[r1 * b + batch] * rhs_opp[c1 * b + batch];
                    v12 += lhs_opp[r1 * b + batch] * rhs_opp[c2 * b + batch];
                    v13 += lhs_opp[r1 * b + batch] * rhs_opp[c3 * b + batch];
                    v14 += lhs_opp[r1 * b + batch] * rhs_opp[c4 * b + batch];

                    v21 += lhs_opp[r2 * b + batch] * rhs_opp[c1 * b + batch];
                    v22 += lhs_opp[r2 * b + batch] * rhs_opp[c2 * b + batch];
                    v23 += lhs_opp[r2 * b + batch] * rhs_opp[c3 * b + batch];
                    v24 += lhs_opp[r2 * b + batch] * rhs_opp[c4 * b + batch];

                    batch += 1;
                }

                out[(r1 - m_start) * n + c1] = v11;
                out[(r1 - m_start) * n + c2] = v12;
                out[(r1 - m_start) * n + c3] = v13;
                out[(r1 - m_start) * n + c4] = v14;

                out[(r2 - m_start) * n + c1] = v21;
                out[(r2 - m_start) * n + c2] = v22;
                out[(r2 - m_start) * n + c3] = v23;
                out[(r2 - m_start) * n + c4] = v24;

                column += 4;
            }

            while column + 1 < n {
                let c1 = column;
                let c2 = column + 1;

                let mut xmm1 = Simd::<T, 8>::splat(T::default());
                let mut xmm2 = Simd::<T, 8>::splat(T::default());
                let mut xmm3 = Simd::<T, 8>::splat(T::default());
                let mut xmm4 = Simd::<T, 8>::splat(T::default());

                let mut batch = 0;

                while batch + lanes - 1 < b {
                    let l1 = Simd::<T, 8>::from_slice(&lhs_opp[r1 * b + batch..]);
                    let l2 = Simd::<T, 8>::from_slice(&lhs_opp[r2 * b + batch..]);

                    let r1 = Simd::<T, 8>::from_slice(&rhs_opp[c1 * b + batch..]);
                    let r2 = Simd::<T, 8>::from_slice(&rhs_opp[c2 * b + batch..]);

                    xmm1 += l1 * r1;
                    xmm2 += l1 * r2;

                    xmm3 += l2 * r1;
                    xmm4 += l2 * r2;

                    batch += lanes;
                }

                let mut v11 = Self::sum(&xmm1.to_array());
                let mut v12 = Self::sum(&xmm2.to_array());

                let mut v21 = Self::sum(&xmm3.to_array());
                let mut v22 = Self::sum(&xmm4.to_array());

                while batch < b {
                    v11 += lhs_opp[r1 * b + batch] * rhs_opp[c1 * b + batch];
                    v12 += lhs_opp[r1 * b + batch] * rhs_opp[c2 * b + batch];

                    v21 += lhs_opp[r2 * b + batch] * rhs_opp[c1 * b + batch];
                    v22 += lhs_opp[r2 * b + batch] * rhs_opp[c2 * b + batch];

                    batch += 1;
                }

                out[(r1 - m_start) * n + c1] = v11;
                out[(r1 - m_start) * n + c2] = v12;

                out[(r2 - m_start) * n + c1] = v21;
                out[(r2 - m_start) * n + c2] = v22;

                column += 2;
            }

            if column < n {
                let mut v1 = T::default();
                let mut v2 = T::default();

                for batch in 0..b {
                    v1 += lhs_opp[r1 * b + batch] * rhs_opp[column * b + batch];
                    v2 += lhs_opp[r2 * b + batch] * rhs_opp[column * b + batch];
                }

                out[(r1 - m_start) * n + column] = v1;
                out[(r2 - m_start) * n + column] = v2;
            }

            row += 2;
        }

        if row < m_end {
            for column in 0..n {
                let mut v = T::default();

                for batch in 0..b {
                    v += lhs_opp[row * b + batch] * rhs_opp[column * b + batch];
                }

                out[(row - m_start) * n + column] = v;
            }
        }
    }

    fn compute_batch_outer_impl(&self, output: &mut Vec<T>) {
        if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 {
            let m = self.lhs.value.columns();
            let n = self.rhs.value.columns();
            let b = self.lhs.value.rows();

            if m * n <= 16384 {
                let small_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| Self::small_kernel(m, n, b, out, lhs, rhs);
                forward_data_binary(output, &self.lhs.value, &self.rhs.value, small_kernel);
            } else {
                let mut rhs_opp = Vec::<T>::new();
                let mut lhs_opp = Vec::<T>::new();

                let mut transpose_first_kernel = |_out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                    lhs_opp = lhs.clone();
                    for lhs_row in 0..b {
                        for lhs_column in 0..m {
                            lhs_opp[lhs_column * b + lhs_row] = lhs[lhs_row * m + lhs_column];
                        }
                    }

                    rhs_opp = rhs.clone();
                    for rhs_row in 0..b {
                        for rhs_column in 0..n {
                            rhs_opp[rhs_column * b + rhs_row] = rhs[rhs_row * n + rhs_column];
                        }
                    }
                };

                forward_data_binary_mut(output, &self.lhs.value, &self.rhs.value, &mut transpose_first_kernel);

                let transposed_kernel = |out: &mut [T], m_start: usize, m_end: usize| {
                    Self::transposed_kernel(m_start, m_end, n, b, out, &lhs_opp, &rhs_opp);
                };
                dispatch_parallel_2d(output, m, n > 20, n, transposed_kernel);
            }
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

// BatchOuterExpr is an EtlExpr
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for BatchOuterExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
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

    fn compute_into_div(&self, output: &mut Vec<T>) {
        self.compute_batch_outer_div(output);
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

// BatchOuterExpr is an EtlWrappable
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for BatchOuterExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
    type WrappedAs = BatchOuterExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// BatchOuterExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for BatchOuterExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
    fn to_data(&self) -> Vec<T> {
        self.temp.clone()
    }
}

// Operations

pub fn batch_outer<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>(
    lhs: LeftExpr,
    rhs: RightExpr,
) -> BatchOuterExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
    BatchOuterExpr::<T, LeftExpr, RightExpr>::new(lhs, rhs)
}

crate::impl_add_op_binary_expr_simd!(BatchOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr_simd!(BatchOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr_simd!(BatchOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr_simd!(BatchOuterExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr_simd!(BatchOuterExpr<T, LeftExpr, RightExpr>);

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

        assert_eq!(c.at2(0, 0), 151);
        assert_eq!(c.at2(0, 1), 160);
        assert_eq!(c.at2(0, 2), 169);
        assert_eq!(c.at2(0, 3), 178);
        assert_eq!(c.at2(1, 0), 196);
        assert_eq!(c.at2(1, 1), 208);
        assert_eq!(c.at2(1, 2), 220);
        assert_eq!(c.at2(1, 3), 232);
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

        assert_eq!(c.at2(0, 0), 302);
        assert_eq!(c.at2(0, 1), 320);
        assert_eq!(c.at2(0, 2), 338);
        assert_eq!(c.at2(0, 3), 356);
        assert_eq!(c.at2(1, 0), 392);
        assert_eq!(c.at2(1, 1), 416);
        assert_eq!(c.at2(1, 2), 440);
        assert_eq!(c.at2(1, 3), 464);
    }

    #[test]
    fn batch_outer_large() {
        let m = 17;
        let n = 11;
        let b = 39;

        let mut lhs = Matrix2d::<i64>::new(b, m);
        let mut rhs = Matrix2d::<i64>::new(b, n);

        lhs.iota_fill(1);
        rhs.iota_fill(2);

        let mut c = Matrix2d::<i64>::new(m, n);
        c |= batch_outer(&lhs, &rhs);

        let mut c_ref = Matrix2d::<i64>::new(m, n);

        for row in 0..m {
            for column in 0..n {
                let mut v = 0;

                for batch in 0..b {
                    v += lhs.at2(batch, row) * rhs.at2(batch, column);
                }

                *c_ref.at_mut(row, column) = v;
            }
        }

        for i in 0..(m * n) {
            assert_eq!(c.at(i), c_ref.at(i), "Invalid value at index {i}");
        }
    }

    #[test]
    fn batch_outer_large_parallel() {
        let m = 171;
        let n = 111;
        let b = 39;

        let mut lhs = Matrix2d::<i64>::new(b, m);
        let mut rhs = Matrix2d::<i64>::new(b, n);

        lhs.iota_fill(1);
        rhs.iota_fill(2);

        let mut c = Matrix2d::<i64>::new(m, n);
        c |= batch_outer(&lhs, &rhs);

        let mut c_ref = Matrix2d::<i64>::new(m, n);

        for row in 0..m {
            for column in 0..n {
                let mut v = 0;

                for batch in 0..b {
                    v += lhs.at2(batch, row) * rhs.at2(batch, column);
                }

                *c_ref.at_mut(row, column) = v;
            }
        }

        for i in 0..(m * n) {
            assert_eq!(c.at(i), c_ref.at(i), "Invalid value at index {i}");
        }
    }
}
