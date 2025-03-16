mod etl;

use crate::etl::vector::Vector;

fn main() {
    let mut vec: Vector = Vector::new(8);

    println!("Hello, world from a Vector({})!", vec.size());

    for (n, value) in vec.iter_mut().enumerate() {
        *value = (n * n) as i64;
    }

    for value in vec.iter() {
        println!("{}", value);
    }
}
