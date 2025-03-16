mod etl;

use crate::etl::vector::Vector;

fn basic<TG: From<i32> + Default + Clone + std::fmt::Display>(size: usize) {
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

fn main() {
    basic::<i64>(8);
    basic::<f64>(8);
}
