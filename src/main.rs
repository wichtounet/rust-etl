mod etl;

use etl::sigmoid_expr::sigmoid;

use crate::etl::matrix_2d::Matrix2d;
use crate::etl::vector::Vector;

use crate::etl::etl_expr::EtlExpr;

struct DenseLayer {
    weights: Matrix2d<f64>,
    biases: Vector<f64>,
}

impl DenseLayer {
    fn new() -> Self {
        Self {
            weights: Matrix2d::<f64>::new(28 * 28, 512),
            biases: Vector::<f64>::new(512),
        }
    }
}

struct Layer {
    weights: Vector<f64>,
    biases: Vector<f64>,
}

impl Layer {
    fn new() -> Self {
        Layer {
            weights: Vector::<f64>::new(1024),
            biases: Vector::<f64>::new(1024),
        }
    }

    fn compute_output(&self, output: &mut Vector<f64>) {
        *output |= sigmoid(&self.weights + &self.biases);
    }
}

fn main() {
    let mut layer = Layer::new();
    let mut dense_layer = DenseLayer::new();

    layer.weights.rand_fill();
    layer.biases.rand_fill();

    dense_layer.weights.rand_fill();
    dense_layer.biases.rand_fill();

    let mut output = Vector::<f64>::new(1024);
    layer.compute_output(&mut output);

    println!("weight: {}", layer.weights.at(8));
    println!("bias:   {}", layer.biases.at(8));
    println!("output: {}", output.at(8));
}
