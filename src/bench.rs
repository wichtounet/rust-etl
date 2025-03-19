mod etl;

use crate::etl::vector::Vector;

use std::time::SystemTime;

fn bench_closure<F: FnMut()>(mut closure: F, rep: usize) -> f64 {
    let now = SystemTime::now();

    for _n in 0..rep {
        closure();
    }

    match now.elapsed() {
       Ok(elapsed) => {
           elapsed.as_millis() as f64 / rep as f64
       }
       Err(e) => {
           panic!("Time Error: {e:?}");
       }
   }
}

fn bench_basic_a(n: usize, r: usize) {
    let a = Vector::<f64>::new_rand(n);
    let b = Vector::<f64>::new_rand(n);
    let mut c = Vector::<f64>::new_rand(n);

    let func = || c|= &a + &b;

    let elapsed = bench_closure(func, r);
    println!("c = a + b ({}) took {}ms", n, elapsed);
}

fn main() {
    bench_basic_a(1024, 128);
    bench_basic_a(8 * 1024, 128);
    bench_basic_a(16 * 1024, 128);
    bench_basic_a(32 * 1024, 128);
    bench_basic_a(64 * 1024, 128);
    bench_basic_a(128 * 1024, 128);
    bench_basic_a(1024 * 1024, 64);
}
