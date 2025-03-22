mod etl;

use crate::etl::matrix_2d::Matrix2d;
use crate::etl::vector::Vector;

use std::time::SystemTime;

fn bench_closure<F: FnMut()>(mut closure: F, rep: usize) -> (f64, f64) {
    let now = SystemTime::now();

    for _n in 0..rep {
        closure();
    }

    match now.elapsed() {
        Ok(elapsed) => (elapsed.as_millis() as f64 / rep as f64, elapsed.as_micros() as f64 / rep as f64),
        Err(e) => {
            panic!("Time Error: {e:?}");
        }
    }
}

fn choose_time(times: (f64, f64)) -> String {
    let (millis, micros) = times;

    if millis <= 0.01 {
        format!("{}us", micros)
    } else {
        format!("{}ms", millis)
    }
}

fn bench_basic_a(n: usize, r: usize) {
    let a = Vector::<f64>::new_rand(n);
    let b = Vector::<f64>::new_rand(n);
    let mut c = Vector::<f64>::new_rand(n);

    let func = || c |= &a + &b;

    let times = bench_closure(func, r);
    println!("c = a + b ({}) took {}", n, choose_time(times));
}

fn bench_basic_b(n: usize, r: usize) {
    let a = Vector::<f64>::new_rand(n);
    let b = Vector::<f64>::new_rand(n);
    let c = Vector::<f64>::new_rand(n);

    let mut d = Vector::<f64>::new_rand(n);

    let func = || d |= &a + &b + &c + &a;

    let times = bench_closure(func, r);
    println!("d = a + b + c + a ({}) took {}", n, choose_time(times));
}

fn bench_gemv(rows: usize, columns: usize, r: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, columns);
    let b = Vector::<f64>::new_rand(columns);
    let mut c = Vector::<f64>::new_rand(rows);

    let func = || c |= &a * &b;

    let times = bench_closure(func, r);
    println!("c = M * v ({}:{}) took {}", rows, columns, choose_time(times));
}

fn main() {
    bench_basic_a(1024, 65536);
    bench_basic_a(8 * 1024, 32768);
    bench_basic_a(16 * 1024, 16384);
    bench_basic_a(32 * 1024, 8192);
    bench_basic_a(64 * 1024, 4096);
    bench_basic_a(128 * 1024, 2048);
    bench_basic_a(1024 * 1024, 1024);

    bench_basic_b(1024, 32768);
    bench_basic_b(8 * 1024, 16384);
    bench_basic_b(16 * 1024, 8192);
    bench_basic_b(32 * 1024, 4096);
    bench_basic_b(64 * 1024, 2048);
    bench_basic_b(128 * 1024, 1024);
    bench_basic_b(1024 * 1024, 512);

    // TODO: Also bench the performanc when combining GEMM and additions or scaling

    bench_gemv(16, 128, 65536);
    bench_gemv(32, 128, 32768);
    bench_gemv(64, 256, 16384);
    bench_gemv(128, 1024, 8192);
    bench_gemv(256, 1024, 4095);
    bench_gemv(1024, 256, 2048);
    bench_gemv(1024, 1024, 1024);
}
