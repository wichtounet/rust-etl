use crate::base_traits::*;
use crate::etl_expr::*;

use std::simd::*;

// The declaration of MulExpr

/// Expression represneting a vector-matrix-multiplication
/// LeftExpr is a vector expression
/// RightExpr is a matrix expression
/// MulExpr is a vector expression
pub struct MulExpr<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>>
where
    Simd<T, 8>: SimdHelper,
{
    lhs: EtlWrapper<T, LeftExpr::WrappedAs>,
    rhs: EtlWrapper<T, RightExpr::WrappedAs>,
    pub temp: Vec<T>,
}

// The functions of MulExpr

impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> MulExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
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

        let mut expr = Self {
            lhs: lhs.wrap(),
            rhs: rhs.wrap(),
            temp: Vec::<T>::new(),
        };

        let mut temp = vec![T::default(); padded_size(expr.size())];
        expr.compute_gemm_impl(&mut temp);
        expr.temp = temp;

        expr
    }

    fn compute_gemm(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        output[..self.temp.len()].copy_from_slice(&self.temp[..]);
    }

    fn compute_gemm_add(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (n, value) in output.iter_mut().enumerate() {
            *value += self.temp[n];
        }
    }

    fn compute_gemm_sub(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (n, value) in output.iter_mut().enumerate() {
            *value -= self.temp[n];
        }
    }

    fn compute_gemm_scale(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (n, value) in output.iter_mut().enumerate() {
            *value *= self.temp[n];
        }
    }

    fn compute_gemm_div(&self, output: &mut [T]) {
        assert!(!self.temp.is_empty());

        for (n, value) in output.iter_mut().enumerate() {
            *value /= self.temp[n];
        }
    }

    fn small_gemm_kernel(m: usize, n: usize, k: usize, out: &mut [T], lhs: &[T], rhs: &[T]) {
        let lanes = 8;

        let mut column = 0;

        // vectorized loop, unrolled twice
        while column + 2 * lanes - 1 < k {
            let column1 = column;
            let column2 = column + lanes;

            let mut row = 0;

            while row + 1 < m {
                let row1 = row;
                let row2 = row + 1;

                // inner = 0
                let mut l1 = Simd::<T, 8>::splat(lhs[row1 * n]);
                let mut l2 = Simd::<T, 8>::splat(lhs[row2 * n]);

                let mut r1 = Simd::<T, 8>::from_slice(&rhs[column1..]);
                let mut r2 = Simd::<T, 8>::from_slice(&rhs[column2..]);

                let mut v1 = l1 * r1;
                let mut v2 = l1 * r2;
                let mut v3 = l2 * r1;
                let mut v4 = l2 * r2;

                for inner in 1..n {
                    l1 = Simd::<T, 8>::splat(lhs[row1 * n + inner]);
                    l2 = Simd::<T, 8>::splat(lhs[row2 * n + inner]);

                    r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column1..]);
                    r2 = Simd::<T, 8>::from_slice(&rhs[inner * k + column2..]);

                    v1 += l1 * r1;
                    v2 += l1 * r2;
                    v3 += l2 * r1;
                    v4 += l2 * r2;
                }

                v1.copy_to_slice(&mut out[row1 * k + column1..row1 * k + column1 + lanes]);
                v2.copy_to_slice(&mut out[row1 * k + column2..row1 * k + column2 + lanes]);
                v3.copy_to_slice(&mut out[row2 * k + column1..row2 * k + column1 + lanes]);
                v4.copy_to_slice(&mut out[row2 * k + column2..row2 * k + column2 + lanes]);

                row += 2;
            }

            if row < m {
                // inner = 0
                let mut l1 = Simd::<T, 8>::splat(lhs[row * n]);

                let mut r1 = Simd::<T, 8>::from_slice(&rhs[column1..]);
                let mut r2 = Simd::<T, 8>::from_slice(&rhs[column2..]);

                let mut v1 = l1 * r1;
                let mut v2 = l1 * r2;

                for inner in 1..n {
                    l1 = Simd::<T, 8>::splat(lhs[row * n + inner]);

                    r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column1..]);
                    r2 = Simd::<T, 8>::from_slice(&rhs[inner * k + column2..]);

                    v1 += l1 * r1;
                    v2 += l1 * r2;
                }

                v1.copy_to_slice(&mut out[row * k + column1..row * k + column1 + lanes]);
                v2.copy_to_slice(&mut out[row * k + column2..row * k + column2 + lanes]);
            }

            column += 2 * lanes;
        }

        // vectorized loop
        while column + lanes - 1 < k {
            let mut row = 0;

            while row + 1 < m {
                let row1 = row;
                let row2 = row + 1;

                // inner = 0
                let mut l1 = Simd::<T, 8>::splat(lhs[row1 * n]);
                let mut l2 = Simd::<T, 8>::splat(lhs[row2 * n]);

                let mut r1 = Simd::<T, 8>::from_slice(&rhs[column..]);

                let mut v1 = l1 * r1;
                let mut v2 = l2 * r1;

                for inner in 1..n {
                    l1 = Simd::<T, 8>::splat(lhs[row1 * n + inner]);
                    l2 = Simd::<T, 8>::splat(lhs[row2 * n + inner]);

                    r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column..]);

                    v1 += l1 * r1;
                    v2 += l2 * r1;
                }

                v1.copy_to_slice(&mut out[row1 * k + column..row1 * k + column + lanes]);
                v2.copy_to_slice(&mut out[row2 * k + column..row2 * k + column + lanes]);

                row += 2;
            }

            if row < m {
                // inner = 0
                let mut l1 = Simd::<T, 8>::splat(lhs[row * n]);
                let mut r1 = Simd::<T, 8>::from_slice(&rhs[column..]);
                let mut v1 = l1 * r1;

                for inner in 1..n {
                    l1 = Simd::<T, 8>::splat(lhs[row * n + inner]);
                    r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column..]);
                    v1 += l1 * r1;
                }

                v1.copy_to_slice(&mut out[row * k + column..row * k + column + lanes]);
            }

            column += lanes;
        }

        while column + 1 < k {
            let c1 = column;
            let c2 = column + 1;

            let mut row = 0;

            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                // inner = 0
                let mut v1 = lhs[r1 * n] * rhs[c1];
                let mut v2 = lhs[r1 * n] * rhs[c2];
                let mut v3 = lhs[r2 * n] * rhs[c1];
                let mut v4 = lhs[r2 * n] * rhs[c2];

                for inner in 1..n {
                    v1 += lhs[r1 * n + inner] * rhs[inner * k + c1];
                    v2 += lhs[r1 * n + inner] * rhs[inner * k + c2];
                    v3 += lhs[r2 * n + inner] * rhs[inner * k + c1];
                    v4 += lhs[r2 * n + inner] * rhs[inner * k + c2];
                }

                out[r1 * k + c1] = v1;
                out[r1 * k + c2] = v2;
                out[r2 * k + c1] = v3;
                out[r2 * k + c2] = v4;

                row += 2;
            }

            if row < m {
                // inner = 0
                let mut v1 = lhs[row * n] * rhs[c1];
                let mut v2 = lhs[row * n] * rhs[c2];

                for inner in 1..n {
                    v1 += lhs[row * n + inner] * rhs[inner * k + c1];
                    v2 += lhs[row * n + inner] * rhs[inner * k + c2];
                }

                out[row * k + c1] = v1;
                out[row * k + c2] = v2;
            }

            column += 2;
        }

        if column < k {
            let mut row = 0;

            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                // inner = 0
                let mut v1 = lhs[r1 * n] * rhs[column];
                let mut v2 = lhs[r2 * n] * rhs[column];

                for inner in 1..n {
                    v1 += lhs[r1 * n + inner] * rhs[inner * k + column];
                    v2 += lhs[r2 * n + inner] * rhs[inner * k + column];
                }

                out[r1 * k + column] = v1;
                out[r2 * k + column] = v2;

                row += 2;
            }

            if row < m {
                // inner = 0
                let mut v = lhs[row * n] * rhs[column];

                for inner in 1..n {
                    v += lhs[row * n + inner] * rhs[inner * k + column];
                }

                out[row * k + column] = v;
            }
        }
    }

    fn medium_gemm_kernel(m: usize, n: usize, k: usize, out: &mut [T], lhs: &[T], rhs: &[T]) {
        let lanes = 8;

        let k_block_size = 128;
        let m_block_size = 64;
        let n_block_size = 128;

        let mut column_start = 0;

        // blocking loop for column
        while column_start < k {
            let column_end = if column_start + k_block_size > k { k } else { column_start + k_block_size };

            let mut row_start = 0;

            // blocking loop for row
            while row_start < m {
                let row_end = if row_start + m_block_size > m { m } else { row_start + m_block_size };

                // Zero out the block
                for column in column_start..column_end {
                    for row in row_start..row_end {
                        out[row * k + column] = T::default();
                    }
                }

                let mut inner_start = 0;

                // blocking loop for inner
                while inner_start < n {
                    let inner_end = if inner_start + n_block_size > n { n } else { inner_start + n_block_size };

                    let mut column = column_start;

                    // vectorized loop, unrolled twice
                    while column + 2 * lanes - 1 < column_end {
                        let column1 = column;
                        let column2 = column + lanes;

                        let mut row = row_start;

                        while row + 1 < row_end {
                            let row1 = row;
                            let row2 = row + 1;

                            let mut v1 = Simd::<T, 8>::from_slice(&out[row1 * k + column1..]);
                            let mut v2 = Simd::<T, 8>::from_slice(&out[row1 * k + column2..]);
                            let mut v3 = Simd::<T, 8>::from_slice(&out[row2 * k + column2..]);
                            let mut v4 = Simd::<T, 8>::from_slice(&out[row2 * k + column2..]);

                            for inner in inner_start..inner_end {
                                let l1 = Simd::<T, 8>::splat(lhs[row1 * n + inner]);
                                let l2 = Simd::<T, 8>::splat(lhs[row2 * n + inner]);

                                let r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column1..]);
                                let r2 = Simd::<T, 8>::from_slice(&rhs[inner * k + column2..]);

                                v1 += l1 * r1;
                                v2 += l1 * r2;
                                v3 += l2 * r1;
                                v4 += l2 * r2;
                            }

                            out[row1 * k + column1..row1 * k + column1 + lanes].copy_from_slice(&v1.to_array());
                            out[row1 * k + column2..row1 * k + column2 + lanes].copy_from_slice(&v2.to_array());
                            out[row2 * k + column1..row2 * k + column1 + lanes].copy_from_slice(&v3.to_array());
                            out[row2 * k + column2..row2 * k + column2 + lanes].copy_from_slice(&v4.to_array());

                            row += 2;
                        }

                        if row < row_end {
                            let mut v1 = Simd::<T, 8>::from_slice(&out[row * k + column1..]);
                            let mut v2 = Simd::<T, 8>::from_slice(&out[row * k + column2..]);

                            for inner in inner_start..inner_end {
                                let l1 = Simd::<T, 8>::splat(lhs[row * n + inner]);

                                let r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column1..]);
                                let r2 = Simd::<T, 8>::from_slice(&rhs[inner * k + column2..]);

                                v1 += l1 * r1;
                                v2 += l1 * r2;
                            }

                            out[row * k + column1..row * k + column1 + lanes].copy_from_slice(&v1.to_array());
                            out[row * k + column2..row * k + column2 + lanes].copy_from_slice(&v2.to_array());
                        }

                        column += 2 * lanes;
                    }

                    // vectorized loop
                    while column + lanes - 1 < column_end {
                        for row in row_start..row_end {
                            let mut v1 = Simd::<T, 8>::from_slice(&out[row * k + column..]);

                            for inner in inner_start..inner_end {
                                let l1 = Simd::<T, 8>::splat(lhs[row * n + inner]);
                                let r1 = Simd::<T, 8>::from_slice(&rhs[inner * k + column..]);
                                v1 += l1 * r1;
                            }

                            out[row * k + column..row * k + column + lanes].copy_from_slice(&v1.to_array());
                        }

                        column += lanes;
                    }

                    // remainder loop
                    while column < column_end {
                        for row in row_start..row_end {
                            let mut v = out[row * k + column];
                            for inner in inner_start..inner_end {
                                v += lhs[row * n + inner] * rhs[inner * k + column];
                            }
                            out[row * k + column] = v;
                        }

                        column += 1;
                    }

                    inner_start += n_block_size;
                }

                row_start += m_block_size;
            }

            column_start += k_block_size;
        }
    }

    fn prev_block(value: usize, lanes: usize) -> usize {
        value - (value % lanes)
    }

    // Note: Cannot use reduce_sum in generic code
    fn sum(input: &[T; 8]) -> T {
        input[0] + input[1] + input[2] + input[3] + input[4] + input[5] + input[6] + input[7]
    }

    // Multiply LHS[m, n] with RHS[n, k] into OUT[m, k]
    fn large_gemm_kernel(column_first: usize, column_last: usize, rows: usize, inner_size: usize, columns: usize, out: &mut [T], lhs: &[T], rhs: &[T]) {
        let lanes = 8;

        let inner_block_size = 112 * (16 / 4); // Optimized for f32
        let column_block_size = 96;

        let mut lhs2 = vec![T::default(); rows * inner_block_size]; // [rows, inner_block_size] (RM)
        let mut rhs2 = vec![T::default(); column_block_size * inner_block_size]; // [inner_block_size, column_block_size] (CM)

        let mut inner_block_index = 0;

        while inner_block_index + lanes - 1 < inner_size {
            let inner_block = if inner_block_index + inner_block_size <= inner_size {
                inner_block_size
            } else {
                Self::prev_block(inner_size - inner_block_index, lanes)
            };

            assert!(inner_block > 0);

            // Copy (standard) lhs -> lhs2
            for sub_row in 0..rows {
                // Copy one column at a time (slightly faster, without bounds check)
                lhs2[sub_row * inner_block_size..sub_row * inner_block_size + inner_block]
                    .copy_from_slice(&lhs[sub_row * inner_size + inner_block_index..sub_row * inner_size + inner_block_index + inner_block]);
            }

            let mut column_block_index = column_first;

            while column_block_index < column_last {
                let column_block = if column_block_index + column_block_size <= column_last {
                    column_block_size
                } else {
                    column_last - column_block_index
                };

                // Copy (transposed) rhs -> rhs2
                for sub_row in 0..inner_block {
                    for sub_column in 0..column_block {
                        rhs2[sub_column * inner_block_size + sub_row] = rhs[(sub_row + inner_block_index) * columns + sub_column + column_block_index];
                    }
                }

                let mut row = 0;

                while row + 3 < rows {
                    let row1 = row;
                    let row2 = row + 1;
                    let row3 = row + 2;
                    let row4 = row + 3;

                    let mut column = 0;

                    while column + 1 < column_block {
                        let column1 = column;
                        let column2 = column + 1;

                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size..]);
                        let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size..]);
                        let l3 = Simd::<T, 8>::from_slice(&lhs2[row3 * inner_block_size..]);
                        let l4 = Simd::<T, 8>::from_slice(&lhs2[row4 * inner_block_size..]);

                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size..]);
                        let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size..]);

                        let mut v1 = l1 * r1;
                        let mut v2 = l2 * r1;
                        let mut v3 = l3 * r1;
                        let mut v4 = l4 * r1;
                        let mut v5 = l1 * r2;
                        let mut v6 = l2 * r2;
                        let mut v7 = l3 * r2;
                        let mut v8 = l4 * r2;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size + inner..]);
                            let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size + inner..]);
                            let l3 = Simd::<T, 8>::from_slice(&lhs2[row3 * inner_block_size + inner..]);
                            let l4 = Simd::<T, 8>::from_slice(&lhs2[row4 * inner_block_size + inner..]);

                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size + inner..]);
                            let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size + inner..]);

                            v1 += l1 * r1;
                            v2 += l2 * r1;
                            v3 += l3 * r1;
                            v4 += l4 * r1;
                            v5 += l1 * r2;
                            v6 += l2 * r2;
                            v7 += l3 * r2;
                            v8 += l4 * r2;
                        }

                        out[row1 * columns + column1 + column_block_index] += Self::sum(v1.as_array());
                        out[row2 * columns + column1 + column_block_index] += Self::sum(v2.as_array());
                        out[row3 * columns + column1 + column_block_index] += Self::sum(v3.as_array());
                        out[row4 * columns + column1 + column_block_index] += Self::sum(v4.as_array());
                        out[row1 * columns + column2 + column_block_index] += Self::sum(v5.as_array());
                        out[row2 * columns + column2 + column_block_index] += Self::sum(v6.as_array());
                        out[row3 * columns + column2 + column_block_index] += Self::sum(v7.as_array());
                        out[row4 * columns + column2 + column_block_index] += Self::sum(v8.as_array());

                        column += 2;
                    }

                    if column < column_block {
                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size..]);
                        let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size..]);
                        let l3 = Simd::<T, 8>::from_slice(&lhs2[row3 * inner_block_size..]);
                        let l4 = Simd::<T, 8>::from_slice(&lhs2[row4 * inner_block_size..]);

                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size..]);

                        let mut v1 = l1 * r1;
                        let mut v2 = l2 * r1;
                        let mut v3 = l3 * r1;
                        let mut v4 = l4 * r1;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size + inner..]);
                            let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size + inner..]);
                            let l3 = Simd::<T, 8>::from_slice(&lhs2[row3 * inner_block_size + inner..]);
                            let l4 = Simd::<T, 8>::from_slice(&lhs2[row4 * inner_block_size + inner..]);

                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size + inner..]);

                            v1 += l1 * r1;
                            v2 += l2 * r1;
                            v3 += l3 * r1;
                            v4 += l4 * r1;
                        }

                        out[row1 * columns + column + column_block_index] += Self::sum(v1.as_array());
                        out[row2 * columns + column + column_block_index] += Self::sum(v2.as_array());
                        out[row3 * columns + column + column_block_index] += Self::sum(v3.as_array());
                        out[row4 * columns + column + column_block_index] += Self::sum(v4.as_array());
                    }

                    row += 4;
                }

                while row + 1 < rows {
                    let row1 = row;
                    let row2 = row + 1;

                    let mut column = 0;

                    while column + 1 < column_block {
                        let column1 = column;
                        let column2 = column + 1;

                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size..]);
                        let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size..]);

                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size..]);
                        let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size..]);

                        let mut v1 = l1 * r1;
                        let mut v2 = l2 * r1;
                        let mut v3 = l1 * r2;
                        let mut v4 = l2 * r2;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size + inner..]);
                            let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size + inner..]);

                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size + inner..]);
                            let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size + inner..]);

                            v1 += l1 * r1;
                            v2 += l2 * r1;
                            v3 += l1 * r2;
                            v4 += l2 * r2;
                        }

                        out[row1 * columns + column1 + column_block_index] += Self::sum(v1.as_array());
                        out[row2 * columns + column1 + column_block_index] += Self::sum(v2.as_array());
                        out[row1 * columns + column2 + column_block_index] += Self::sum(v3.as_array());
                        out[row2 * columns + column2 + column_block_index] += Self::sum(v4.as_array());

                        column += 2;
                    }

                    if column < column_block {
                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size..]);
                        let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size..]);

                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size..]);

                        let mut v1 = l1 * r1;
                        let mut v2 = l2 * r1;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row1 * inner_block_size + inner..]);
                            let l2 = Simd::<T, 8>::from_slice(&lhs2[row2 * inner_block_size + inner..]);

                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size + inner..]);

                            v1 += l1 * r1;
                            v2 += l2 * r1;
                        }

                        out[row1 * columns + column + column_block_index] += Self::sum(v1.as_array());
                        out[row2 * columns + column + column_block_index] += Self::sum(v2.as_array());
                    }

                    row += 2;
                }

                while row < rows {
                    let mut column = 0;

                    while column + 1 < column_block {
                        let column1 = column;
                        let column2 = column + 1;

                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row * inner_block_size..]);

                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size..]);
                        let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size..]);

                        let mut v1 = l1 * r1;
                        let mut v2 = l1 * r2;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row * inner_block_size + inner..]);

                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column1 * inner_block_size + inner..]);
                            let r2 = Simd::<T, 8>::from_slice(&rhs2[column2 * inner_block_size + inner..]);

                            v1 += l1 * r1;
                            v2 += l1 * r2;
                        }

                        out[row * columns + column1 + column_block_index] += Self::sum(v1.as_array());
                        out[row * columns + column2 + column_block_index] += Self::sum(v2.as_array());

                        column += 2;
                    }

                    if column < column_block {
                        // inner = 0
                        let l1 = Simd::<T, 8>::from_slice(&lhs2[row * inner_block_size..]);
                        let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size..]);
                        let mut v1 = l1 * r1;

                        for inner in (lanes..inner_block).step_by(lanes) {
                            let l1 = Simd::<T, 8>::from_slice(&lhs2[row * inner_block_size + inner..]);
                            let r1 = Simd::<T, 8>::from_slice(&rhs2[column * inner_block_size + inner..]);
                            v1 += l1 * r1;
                        }

                        out[row * columns + column + column_block_index] += Self::sum(v1.as_array());
                    }

                    row += 1;
                }

                column_block_index += column_block;
            }

            inner_block_index += inner_block;
        }

        // Remainder loop (incomple blocks)
        if inner_block_index < inner_size {
            let inner_block = inner_size - inner_block_index;

            // Copy lhs -> lhs2
            for sub_row in 0..rows {
                for sub_column in 0..inner_block {
                    lhs2[sub_row * inner_block_size + sub_column] = lhs[sub_row * inner_size + sub_column + inner_block_index];
                }
            }

            let mut column_block_index = column_first;

            while column_block_index < column_last {
                let column_block = if column_block_index + column_block_size <= column_last {
                    column_block_size
                } else {
                    column_last - column_block_index
                };

                // Copy rhs -> rhs2
                for sub_row in 0..inner_block {
                    for sub_column in 0..column_block {
                        rhs2[sub_column * inner_block_size + sub_row] = rhs[(sub_row + inner_block_index) * columns + sub_column + column_block_index];
                    }
                }

                let mut row = 0;

                // TODO Can probably still unroll more here

                while row < rows {
                    let mut column = 0;

                    while column < column_block {
                        for inner in 0..inner_block {
                            out[row * columns + column + column_block_index] += lhs2[row * inner_block_size + inner] * rhs2[column * inner_block_size + inner];
                        }

                        column += 1;
                    }

                    row += 1;
                }

                column_block_index += column_block;
            }
        }
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

            if n * m < 100 * 100 {
                let small_gemm_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| Self::small_gemm_kernel(m, n, k, out, lhs, rhs);
                forward_data_binary(output, &self.lhs.value, &self.rhs.value, small_gemm_kernel);
            } else if n * m < 200 * 200 {
                let medium_gemm_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| Self::medium_gemm_kernel(m, n, k, out, lhs, rhs);
                forward_data_binary(output, &self.lhs.value, &self.rhs.value, medium_gemm_kernel);
            } else {
                // the forwarding kernel
                let large_gemm_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| {
                    // The kernel for a single thread
                    let large_gemm_kernel_thread =
                        |par_out: &mut [T], first: usize, last: usize| Self::large_gemm_kernel(first, last, m, n, k, par_out, lhs, rhs);
                    dispatch_parallel_block(out, k, 96, large_gemm_kernel_thread);
                };
                forward_data_binary(output, &self.lhs.value, &self.rhs.value, large_gemm_kernel);
            }
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
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlExpr<T> for MulExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
    const DIMENSIONS: usize = if LeftExpr::DIMENSIONS == 2 && RightExpr::DIMENSIONS == 2 { 2 } else { 1 };
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

    fn compute_into_div(&self, output: &mut Vec<T>) {
        self.compute_gemm_div(output);
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

// MulExpr is an EtlWrappable
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlWrappable<T> for MulExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
    type WrappedAs = MulExpr<T, LeftExpr, RightExpr>;

    fn wrap(self) -> EtlWrapper<T, Self::WrappedAs> {
        EtlWrapper {
            value: self,
            _marker: std::marker::PhantomData,
        }
    }
}

// MulExpr computes as copy
impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>> EtlComputable<T> for MulExpr<T, LeftExpr, RightExpr>
where
    Simd<T, 8>: SimdHelper,
{
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
        impl<'a, T: EtlValueType, RightExpr: WrappableExpr<T>> std::ops::Mul<RightExpr> for &'a $type
        where
            std::simd::Simd<T, 8>: $crate::base_traits::SimdHelper,
        {
            type Output = $crate::mul_expr::MulExpr<T, &'a $type, RightExpr>;

            fn mul(self, other: RightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_binary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type
        where
            std::simd::Simd<T, 8>: $crate::base_traits::SimdHelper,
        {
            type Output = $crate::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

// TODO We can remove this macro soon
#[macro_export]
macro_rules! impl_mul_op_binary_expr_simd {
    ($type:ty) => {
        impl<T: EtlValueType, LeftExpr: WrappableExpr<T>, RightExpr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type
        where
            std::simd::Simd<T, 8>: $crate::base_traits::SimdHelper,
        {
            type Output = $crate::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_unary_expr {
    ($type:ty) => {
        impl<T: EtlValueType, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type
        where
            std::simd::Simd<T, 8>: $crate::base_traits::SimdHelper,
        {
            type Output = $crate::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul_op_unary_expr_trait {
    ($trait:tt, $type:ty) => {
        impl<T: EtlValueType + $trait, Expr: WrappableExpr<T>, OuterRightExpr: WrappableExpr<T>> std::ops::Mul<OuterRightExpr> for $type
        where
            std::simd::Simd<T, 8>: $crate::base_traits::SimdHelper,
        {
            type Output = $crate::mul_expr::MulExpr<T, $type, OuterRightExpr>;

            fn mul(self, other: OuterRightExpr) -> Self::Output {
                Self::Output::new(self, other)
            }
        }
    };
}

crate::impl_add_op_binary_expr_simd!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_sub_op_binary_expr_simd!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_mul_op_binary_expr_simd!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_div_op_binary_expr_simd!(MulExpr<T, LeftExpr, RightExpr>);
crate::impl_scale_op_binary_expr_simd!(MulExpr<T, LeftExpr, RightExpr>);

// The tests

#[cfg(test)]
mod tests {
    use crate::constant::cst;
    use crate::etl_expr::EtlExpr;
    use crate::matrix_2d::Matrix2d;
    use crate::vector::Vector;

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

    #[test]
    fn gemm_large() {
        let m = 17;
        let n = 11;
        let k = 39;

        let mut lhs = Matrix2d::<i64>::new(m, n);
        let mut rhs = Matrix2d::<i64>::new(n, k);

        lhs.iota_fill(1);
        rhs.iota_fill(2);

        let mut c = Matrix2d::<i64>::new(m, k);
        c |= &lhs * &rhs;

        let mut c_ref = Matrix2d::<i64>::new(m, k);

        for row in 0..m {
            for column in 0..k {
                let mut v = 0;
                for inner in 0..n {
                    v += lhs.at2(row, inner) * rhs.at2(inner, column);
                }
                *c_ref.at_mut(row, column) = v;
            }
        }

        for i in 0..(m * k) {
            assert_eq!(c.at(i), c_ref.at(i), "Invalid value at index {i}");
        }
    }

    #[test]
    fn gemm_large_parallel() {
        let m = 371;
        let n = 311;
        let k = 39;

        let mut lhs = Matrix2d::<i64>::new(m, n);
        let mut rhs = Matrix2d::<i64>::new(n, k);

        lhs.iota_fill(1);
        rhs.iota_fill(2);

        let mut c = Matrix2d::<i64>::new(m, k);
        c |= &lhs * &rhs;

        let mut c_ref = Matrix2d::<i64>::new(m, k);

        for row in 0..m {
            for column in 0..k {
                let mut v = 0;
                for inner in 0..n {
                    v += lhs.at2(row, inner) * rhs.at2(inner, column);
                }
                *c_ref.at_mut(row, column) = v;
            }
        }

        for i in 0..(m * k) {
            assert_eq!(c.at(i), c_ref.at(i), "Invalid value at index {i}");
        }
    }
}
