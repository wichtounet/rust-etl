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

    fn compute_gemm(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        output[..self.temp.len()].copy_from_slice(&self.temp[..]);
    }

    fn compute_gemm_add(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] += self.temp[n];
        }
    }

    fn compute_gemm_sub(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] -= self.temp[n];
        }
    }

    fn compute_gemm_scale(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] *= self.temp[n];
        }
    }

    fn compute_gemm_div(&self, output: &mut Vec<T>) {
        assert!(!self.temp.is_empty());

        for n in 0..self.temp.len() {
            output[n] /= self.temp[n];
        }
    }

    fn small_gemm_kernel(m: usize, n: usize, k: usize, out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>) {
        let lanes = 8;

        let mut column = 0;

        while column + 7 < k {
            let c1 = column;
            let c2 = column + 1;
            let c3 = column + 2;
            let c4 = column + 3;
            let c5 = column + 4;
            let c6 = column + 5;
            let c7 = column + 6;
            let c8 = column + 7;

            let mut row = 0;

            // medium loop unrolled four times
            while row + 3 < m {
                let r1 = row;
                let r2 = row + 1;
                let r3 = row + 2;
                let r4 = row + 3;

                let mut v11 = out[r1 * k + c1];
                let mut v12 = out[r2 * k + c1];
                let mut v13 = out[r3 * k + c1];
                let mut v14 = out[r4 * k + c1];

                let mut v21 = out[r1 * k + c2];
                let mut v22 = out[r2 * k + c2];
                let mut v23 = out[r3 * k + c2];
                let mut v24 = out[r4 * k + c2];

                let mut v31 = out[r1 * k + c3];
                let mut v32 = out[r2 * k + c3];
                let mut v33 = out[r3 * k + c3];
                let mut v34 = out[r4 * k + c3];

                let mut v41 = out[r1 * k + c4];
                let mut v42 = out[r2 * k + c4];
                let mut v43 = out[r3 * k + c4];
                let mut v44 = out[r4 * k + c4];

                let mut v51 = out[r1 * k + c5];
                let mut v52 = out[r2 * k + c5];
                let mut v53 = out[r3 * k + c5];
                let mut v54 = out[r4 * k + c5];

                let mut v61 = out[r1 * k + c6];
                let mut v62 = out[r2 * k + c6];
                let mut v63 = out[r3 * k + c6];
                let mut v64 = out[r4 * k + c6];

                let mut v71 = out[r1 * k + c7];
                let mut v72 = out[r2 * k + c7];
                let mut v73 = out[r3 * k + c7];
                let mut v74 = out[r4 * k + c7];

                let mut v81 = out[r1 * k + c8];
                let mut v82 = out[r2 * k + c8];
                let mut v83 = out[r3 * k + c8];
                let mut v84 = out[r4 * k + c8];

                for inner in 0..n {
                    v11 += lhs[r1 * n + inner] * rhs[inner * k + c1];
                    v12 += lhs[r2 * n + inner] * rhs[inner * k + c1];
                    v13 += lhs[r3 * n + inner] * rhs[inner * k + c1];
                    v14 += lhs[r4 * n + inner] * rhs[inner * k + c1];

                    v21 += lhs[r1 * n + inner] * rhs[inner * k + c2];
                    v22 += lhs[r2 * n + inner] * rhs[inner * k + c2];
                    v23 += lhs[r3 * n + inner] * rhs[inner * k + c2];
                    v24 += lhs[r4 * n + inner] * rhs[inner * k + c2];

                    v31 += lhs[r1 * n + inner] * rhs[inner * k + c3];
                    v32 += lhs[r2 * n + inner] * rhs[inner * k + c3];
                    v33 += lhs[r3 * n + inner] * rhs[inner * k + c3];
                    v34 += lhs[r4 * n + inner] * rhs[inner * k + c3];

                    v41 += lhs[r1 * n + inner] * rhs[inner * k + c4];
                    v42 += lhs[r2 * n + inner] * rhs[inner * k + c4];
                    v43 += lhs[r3 * n + inner] * rhs[inner * k + c4];
                    v44 += lhs[r4 * n + inner] * rhs[inner * k + c4];

                    v51 += lhs[r1 * n + inner] * rhs[inner * k + c5];
                    v52 += lhs[r2 * n + inner] * rhs[inner * k + c5];
                    v53 += lhs[r3 * n + inner] * rhs[inner * k + c5];
                    v54 += lhs[r4 * n + inner] * rhs[inner * k + c5];

                    v61 += lhs[r1 * n + inner] * rhs[inner * k + c6];
                    v62 += lhs[r2 * n + inner] * rhs[inner * k + c6];
                    v63 += lhs[r3 * n + inner] * rhs[inner * k + c6];
                    v64 += lhs[r4 * n + inner] * rhs[inner * k + c6];

                    v71 += lhs[r1 * n + inner] * rhs[inner * k + c7];
                    v72 += lhs[r2 * n + inner] * rhs[inner * k + c7];
                    v73 += lhs[r3 * n + inner] * rhs[inner * k + c7];
                    v74 += lhs[r4 * n + inner] * rhs[inner * k + c7];

                    v81 += lhs[r1 * n + inner] * rhs[inner * k + c8];
                    v82 += lhs[r2 * n + inner] * rhs[inner * k + c8];
                    v83 += lhs[r3 * n + inner] * rhs[inner * k + c8];
                    v84 += lhs[r4 * n + inner] * rhs[inner * k + c8];
                }

                out[r1 * k + c1] = v11;
                out[r2 * k + c1] = v12;
                out[r3 * k + c1] = v13;
                out[r4 * k + c1] = v14;

                out[r1 * k + c2] = v21;
                out[r2 * k + c2] = v22;
                out[r3 * k + c2] = v23;
                out[r4 * k + c2] = v24;

                out[r1 * k + c3] = v31;
                out[r2 * k + c3] = v32;
                out[r3 * k + c3] = v33;
                out[r4 * k + c3] = v34;

                out[r1 * k + c4] = v41;
                out[r2 * k + c4] = v42;
                out[r3 * k + c4] = v43;
                out[r4 * k + c4] = v44;

                out[r1 * k + c5] = v51;
                out[r2 * k + c5] = v52;
                out[r3 * k + c5] = v53;
                out[r4 * k + c5] = v54;

                out[r1 * k + c6] = v61;
                out[r2 * k + c6] = v62;
                out[r3 * k + c6] = v63;
                out[r4 * k + c6] = v64;

                out[r1 * k + c7] = v71;
                out[r2 * k + c7] = v72;
                out[r3 * k + c7] = v73;
                out[r4 * k + c7] = v74;

                out[r1 * k + c8] = v81;
                out[r2 * k + c8] = v82;
                out[r3 * k + c8] = v83;
                out[r4 * k + c8] = v84;

                row += 4;
            }

            // medium loop unrolled twice
            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                let mut v11 = out[r1 * k + c1];
                let mut v12 = out[r2 * k + c1];

                let mut v21 = out[r1 * k + c2];
                let mut v22 = out[r2 * k + c2];

                let mut v31 = out[r1 * k + c3];
                let mut v32 = out[r2 * k + c3];

                let mut v41 = out[r1 * k + c4];
                let mut v42 = out[r2 * k + c4];

                let mut v51 = out[r1 * k + c5];
                let mut v52 = out[r2 * k + c5];

                let mut v61 = out[r1 * k + c6];
                let mut v62 = out[r2 * k + c6];

                let mut v71 = out[r1 * k + c7];
                let mut v72 = out[r2 * k + c7];

                let mut v81 = out[r1 * k + c8];
                let mut v82 = out[r2 * k + c8];

                for inner in 0..n {
                    v11 += lhs[r1 * n + inner] * rhs[inner * k + c1];
                    v12 += lhs[r2 * n + inner] * rhs[inner * k + c1];

                    v21 += lhs[r1 * n + inner] * rhs[inner * k + c2];
                    v22 += lhs[r2 * n + inner] * rhs[inner * k + c2];

                    v31 += lhs[r1 * n + inner] * rhs[inner * k + c3];
                    v32 += lhs[r2 * n + inner] * rhs[inner * k + c3];

                    v41 += lhs[r1 * n + inner] * rhs[inner * k + c4];
                    v42 += lhs[r2 * n + inner] * rhs[inner * k + c4];

                    v51 += lhs[r1 * n + inner] * rhs[inner * k + c5];
                    v52 += lhs[r2 * n + inner] * rhs[inner * k + c5];

                    v61 += lhs[r1 * n + inner] * rhs[inner * k + c6];
                    v62 += lhs[r2 * n + inner] * rhs[inner * k + c6];

                    v71 += lhs[r1 * n + inner] * rhs[inner * k + c7];
                    v72 += lhs[r2 * n + inner] * rhs[inner * k + c7];

                    v81 += lhs[r1 * n + inner] * rhs[inner * k + c8];
                    v82 += lhs[r2 * n + inner] * rhs[inner * k + c8];
                }

                out[r1 * k + c1] = v11;
                out[r2 * k + c1] = v12;

                out[r1 * k + c2] = v21;
                out[r2 * k + c2] = v22;

                out[r1 * k + c3] = v31;
                out[r2 * k + c3] = v32;

                out[r1 * k + c4] = v41;
                out[r2 * k + c4] = v42;

                out[r1 * k + c5] = v51;
                out[r2 * k + c5] = v52;

                out[r1 * k + c6] = v61;
                out[r2 * k + c6] = v62;

                out[r1 * k + c7] = v71;
                out[r2 * k + c7] = v72;

                out[r1 * k + c8] = v81;
                out[r2 * k + c8] = v82;

                row += 2;
            }

            // medium loop remainder
            if row < m {
                let mut v1 = out[row * k + c1];
                let mut v2 = out[row * k + c2];
                let mut v3 = out[row * k + c3];
                let mut v4 = out[row * k + c4];
                let mut v5 = out[row * k + c5];
                let mut v6 = out[row * k + c6];
                let mut v7 = out[row * k + c7];
                let mut v8 = out[row * k + c8];

                for inner in 0..n {
                    v1 += lhs[row * n + inner] * rhs[inner * k + c1];
                    v2 += lhs[row * n + inner] * rhs[inner * k + c2];
                    v3 += lhs[row * n + inner] * rhs[inner * k + c3];
                    v4 += lhs[row * n + inner] * rhs[inner * k + c4];
                    v5 += lhs[row * n + inner] * rhs[inner * k + c5];
                    v6 += lhs[row * n + inner] * rhs[inner * k + c6];
                    v7 += lhs[row * n + inner] * rhs[inner * k + c7];
                    v8 += lhs[row * n + inner] * rhs[inner * k + c8];
                }

                out[row * k + c1] = v1;
                out[row * k + c2] = v2;
                out[row * k + c3] = v3;
                out[row * k + c4] = v4;
                out[row * k + c5] = v5;
                out[row * k + c6] = v6;
                out[row * k + c7] = v7;
                out[row * k + c8] = v8;
            }

            column += 8;
        }

        while column + 3 < k {
            let c1 = column;
            let c2 = column + 1;
            let c3 = column + 2;
            let c4 = column + 3;

            let mut row = 0;

            // medium loop unrolled twice
            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                let mut v11 = out[r1 * k + c1];
                let mut v12 = out[r2 * k + c1];

                let mut v21 = out[r1 * k + c2];
                let mut v22 = out[r2 * k + c2];

                let mut v31 = out[r1 * k + c3];
                let mut v32 = out[r2 * k + c3];

                let mut v41 = out[r1 * k + c4];
                let mut v42 = out[r2 * k + c4];

                for inner in 0..n {
                    v11 += lhs[r1 * n + inner] * rhs[inner * k + c1];
                    v12 += lhs[r2 * n + inner] * rhs[inner * k + c1];

                    v21 += lhs[r1 * n + inner] * rhs[inner * k + c2];
                    v22 += lhs[r2 * n + inner] * rhs[inner * k + c2];

                    v31 += lhs[r1 * n + inner] * rhs[inner * k + c3];
                    v32 += lhs[r2 * n + inner] * rhs[inner * k + c3];

                    v41 += lhs[r1 * n + inner] * rhs[inner * k + c4];
                    v42 += lhs[r2 * n + inner] * rhs[inner * k + c4];
                }

                out[r1 * k + c1] = v11;
                out[r2 * k + c1] = v12;

                out[r1 * k + c2] = v21;
                out[r2 * k + c2] = v22;

                out[r1 * k + c3] = v31;
                out[r2 * k + c3] = v32;

                out[r1 * k + c4] = v41;
                out[r2 * k + c4] = v42;

                row += 2;
            }

            // medium loop remainder
            if row < m {
                let mut v1 = out[row * k + c1];
                let mut v2 = out[row * k + c2];
                let mut v3 = out[row * k + c3];
                let mut v4 = out[row * k + c4];

                for inner in 0..n {
                    v1 += lhs[row * n + inner] * rhs[inner * k + c1];
                    v2 += lhs[row * n + inner] * rhs[inner * k + c2];
                    v3 += lhs[row * n + inner] * rhs[inner * k + c3];
                    v4 += lhs[row * n + inner] * rhs[inner * k + c4];
                }

                out[row * k + c1] = v1;
                out[row * k + c2] = v2;
                out[row * k + c3] = v3;
                out[row * k + c4] = v4;
            }

            column += 4;
        }

        while column + 1 < k {
            let c1 = column;
            let c2 = column + 1;

            let mut row = 0;

            // medium loop unrolled twice
            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                let mut v11 = out[r1 * k + c1];
                let mut v12 = out[r2 * k + c1];

                let mut v21 = out[r1 * k + c2];
                let mut v22 = out[r2 * k + c2];

                for inner in 0..n {
                    v11 += lhs[r1 * n + inner] * rhs[inner * k + c1];
                    v12 += lhs[r2 * n + inner] * rhs[inner * k + c1];

                    v21 += lhs[r1 * n + inner] * rhs[inner * k + c2];
                    v22 += lhs[r2 * n + inner] * rhs[inner * k + c2];
                }

                out[r1 * k + c1] = v11;
                out[r2 * k + c1] = v12;

                out[r1 * k + c2] = v21;
                out[r2 * k + c2] = v22;

                row += 2;
            }

            // medium loop remainder
            if row < m {
                let mut v1 = out[row * k + c1];
                let mut v2 = out[row * k + c2];

                for inner in 0..n {
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

            // medium loop unrolled twice
            while row + 1 < m {
                let r1 = row;
                let r2 = row + 1;

                let mut v1 = out[r1 * k + column];
                let mut v2 = out[r2 * k + column];

                for inner in 0..n {
                    v1 += lhs[r1 * n + inner] * rhs[inner * k + column];
                    v2 += lhs[r2 * n + inner] * rhs[inner * k + column];
                }

                out[r1 * k + column] = v1;
                out[r2 * k + column] = v2;

                row += 2;
            }

            // medium loop remainder
            if row < m {
                let mut v = out[row * k + column];

                for inner in 0..n {
                    v += lhs[row * n + inner] * rhs[inner * k + column];
                }

                out[row * k + column] = v;
            }
        }
    }

    fn medium_gemm_kernel(m: usize, n: usize, k: usize, out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>) {
        let k_block_size = 128;
        let m_block_size = 64;
        let n_block_size = 128;

        let mut column_start = 0;

        while column_start < k {
            let column_end = if column_start + k_block_size > k { k } else { column_start + k_block_size };

            let mut row_start = 0;

            while row_start < m {
                let row_end = if row_start + m_block_size > m { m } else { row_start + m_block_size };

                // Zero out the block
                for column in column_start..column_end {
                    for row in row_start..row_end {
                        out[row * k + column] = T::default();
                    }
                }

                let mut inner_start = 0;
                while inner_start < n {
                    let inner_end = if inner_start + n_block_size > n { n } else { inner_start + n_block_size };

                    let mut column = column_start;

                    while column + 3 < column_end {
                        let c1 = column;
                        let c2 = column + 1;
                        let c3 = column + 2;
                        let c4 = column + 3;

                        for row in row_start..row_end {
                            let mut v1 = out[row * k + c2];
                            let mut v2 = out[row * k + c1];
                            let mut v3 = out[row * k + c3];
                            let mut v4 = out[row * k + c4];

                            for inner in inner_start..inner_end {
                                v1 += lhs[row * n + inner] * rhs[inner * k + c1];
                                v2 += lhs[row * n + inner] * rhs[inner * k + c2];
                                v3 += lhs[row * n + inner] * rhs[inner * k + c3];
                                v4 += lhs[row * n + inner] * rhs[inner * k + c4];
                            }

                            out[row * k + c1] = v1;
                            out[row * k + c2] = v2;
                            out[row * k + c3] = v3;
                            out[row * k + c4] = v4;
                        }

                        column += 4;
                    }

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

            let small_gemm_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| Self::small_gemm_kernel(m, n, k, out, lhs, rhs);

            // The medium kernel is WIP
            let _medium_gemm_kernel = |out: &mut Vec<T>, lhs: &Vec<T>, rhs: &Vec<T>| Self::medium_gemm_kernel(m, n, k, out, lhs, rhs);

            forward_data_binary(output, &self.lhs.value, &self.rhs.value, small_gemm_kernel);
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
        let m = 171;
        let n = 111;
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
