mod etl;

use crate::etl::vector::Vector;

use crate::etl::etl_expr::EtlExpr;

struct Layer {
    weights: Vector<f64>,
    biases: Vector<f64>
}

impl Layer {
    fn new() -> Self {
        Layer { weights: Vector::<f64>::new(1024), biases: Vector::<f64>::new(1024) }
    }

    fn compute_output(&self, output: &mut Vector::<f64>) {
        output.assign(&self.weights + &self.biases);
    }
}

fn main() {
    let mut layer = Layer::new();

    layer.weights.rand_fill();
    layer.biases.rand_fill();

    let mut output = Vector::<f64>::new(1024);
    layer.compute_output(&mut output);

    println!("{}", output.at(8));
}
