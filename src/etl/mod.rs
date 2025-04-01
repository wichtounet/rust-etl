// The expressions
pub mod add_expr;
pub mod batch_outer_expr;
pub mod bias_add_expr;
pub mod bias_batch_sum_expr;
pub mod etl_expr;
pub mod exp_expr;
pub mod mul_expr;
pub mod scale_expr;
pub mod sigmoid_derivative_expr;
pub mod sigmoid_expr;
pub mod softmax_expr;
pub mod sub_expr;

// The containers
pub mod matrix_2d;
pub mod vector;

// The pseudo containers
pub mod constant;

// Free functions
pub mod reductions;
