mod etl;

use crate::etl::vector::Vector;

use crate::etl::etl_expr::EtlExpr;
use crate::etl::etl_expr::EtlValueType;

fn basic<TG: From<i32> + EtlValueType + std::fmt::Display>(size: usize) {
    let mut vec: Vector<TG> = Vector::<TG>::new(size);

    println!("Hello, world from a Vector<{}>({})!", std::any::type_name::<TG>(), vec.size());

    for (n, value) in vec.iter_mut().enumerate() {
        let mut v: i32 = n as i32;
        v *= v;
        *value = v.into();
    }

    for value in vec.iter() {
        println!("{}", value);
    }
}

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
    basic::<i64>(8);
    basic::<f64>(8);

    let layer = Layer::new();
    let mut output = Vector::<f64>::new(1024);
    layer.compute_output(&mut output);
}
