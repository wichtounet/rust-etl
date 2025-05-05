use etl::batch_outer_expr::batch_outer;
use etl::bias_add_expr::bias_add;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

use std::time::SystemTime;

fn choose_repeat<F: FnMut()>(mut closure: F) -> i64 {
    let now = SystemTime::now();

    for _n in 0..2 {
        closure();
    }

    let elapsed = match now.elapsed() {
        Ok(elapsed) => elapsed.as_nanos() as f64 / 2.0,
        Err(e) => {
            panic!("Time Error: {e:?}");
        }
    };

    let rep = (2000000000.0 / elapsed) as i64;
    if rep == 0 {
        return 1;
    }
    rep
}

fn bench_closure<F: FnMut()>(mut closure: F) -> (f64, f64) {
    let rep = choose_repeat(|| closure());

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

fn bench_basic_a(n: usize) {
    let a = Vector::<f64>::new_rand(n);
    let b = Vector::<f64>::new_rand(n);
    let mut c = Vector::<f64>::new_rand(n);

    let func = || c |= &a + &b;

    let times = bench_closure(func);
    println!("c = a + b ({}) took {}", n, choose_time(times));
}

fn bench_basic_b(n: usize) {
    let a = Vector::<f64>::new_rand(n);
    let b = Vector::<f64>::new_rand(n);
    let c = Vector::<f64>::new_rand(n);

    let mut d = Vector::<f64>::new_rand(n);

    let func = || d |= &a >> &b + &c >> &a;

    let times = bench_closure(func);
    println!("d = a >> b + c >> a ({}) took {}", n, choose_time(times));
}

fn bench_gemv(rows: usize, columns: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, columns);
    let b = Vector::<f64>::new_rand(columns);
    let mut c = Vector::<f64>::new_rand(rows);

    let func = || c |= &a * &b;

    let times = bench_closure(func);
    println!("c = M * v ({}:{}) took {}", rows, columns, choose_time(times));
}

fn bench_gemm(rows: usize, columns: usize, inner: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, inner);
    let b = Matrix2d::<f64>::new_rand(inner, columns);
    let mut c = Matrix2d::<f64>::new_rand(rows, columns);

    let func = || c |= &a * &b;

    let times = bench_closure(func);
    println!("c = A * B ({}:{}:{}) took {}", rows, inner, columns, choose_time(times));
}

fn bench_gemm_outer(rows: usize, columns: usize, inner: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, inner);
    let b = Matrix2d::<f64>::new_rand(inner, columns);
    let x = Matrix2d::<f64>::new_rand(rows, columns);

    let mut c = Matrix2d::<f64>::new_rand(rows, columns);

    let func = || c |= &x + (&a * &b) + &x;

    let times = bench_closure(func);
    println!("c = X + (A * B) + X ({}:{}:{}) took {}", rows, inner, columns, choose_time(times));
}

fn bench_gemm_inner(rows: usize, columns: usize, inner: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, inner);
    let x = Matrix2d::<f64>::new_rand(rows, inner);

    let b = Matrix2d::<f64>::new_rand(inner, columns);
    let y = Matrix2d::<f64>::new_rand(inner, columns);

    let mut c = Matrix2d::<f64>::new_rand(rows, columns);

    let func = || c |= (&a + &x) * (&y + &b);

    let times = bench_closure(func);
    println!("c = (A + X) * (X + B) ({}:{}:{}) took {}", rows, inner, columns, choose_time(times));
}

fn bench_gemm_chain(rows: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, rows);
    let b = Matrix2d::<f64>::new_rand(rows, rows);
    let c = Matrix2d::<f64>::new_rand(rows, rows);
    let d = Matrix2d::<f64>::new_rand(rows, rows);

    let mut y = Matrix2d::<f64>::new_rand(rows, rows);

    let func = || y |= &a * (&b * (&c * &d));

    let times = bench_closure(func);
    println!("y = A * (B * (C * D)) ({}:{}) took {}", rows, rows, choose_time(times));
}

fn bench_batch_outer(rows: usize, columns: usize, batch: usize) {
    let a = Matrix2d::<f64>::new_rand(batch, rows);
    let b = Matrix2d::<f64>::new_rand(batch, columns);
    let mut c = Matrix2d::<f64>::new_rand(rows, columns);

    let func = || c |= batch_outer(&a, &b);

    let times = bench_closure(func);
    println!("c = batch_outer(A, B) ({}:{}:{}) took {}", rows, columns, batch, choose_time(times));
}

fn bench_bias_add(rows: usize, columns: usize) {
    let a = Matrix2d::<f64>::new_rand(rows, columns);
    let b = Vector::<f64>::new_rand(columns);
    let mut c = Matrix2d::<f64>::new_rand(rows, columns);

    let func = || c |= bias_add(&a, &b);

    let times = bench_closure(func);
    println!("c = bias_add(A, B) ({}:{}) took {}", rows, columns, choose_time(times));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let filter = if args.len() > 1 { args[args.len() - 1].clone() } else { "*".to_string() };

    if filter == "*" || filter == "basic" {
        bench_basic_a(1024);
        bench_basic_a(8 * 1024);
        bench_basic_a(16 * 1024);
        bench_basic_a(32 * 1024);
        bench_basic_a(64 * 1024);
        bench_basic_a(128 * 1024);
        bench_basic_a(1024 * 1024);
        bench_basic_a(16 * 1024 * 1024);
        bench_basic_a(32 * 1024 * 1024);

        bench_basic_b(1024);
        bench_basic_b(8 * 1024);
        bench_basic_b(16 * 1024);
        bench_basic_b(32 * 1024);
        bench_basic_b(64 * 1024);
        bench_basic_b(128 * 1024);
        bench_basic_b(1024 * 1024);
        bench_basic_b(16 * 1024 * 1024);
        bench_basic_b(32 * 1024 * 1024);
    }

    if filter == "*" || filter == "gemv" {
        bench_gemv(16, 128);
        bench_gemv(32, 128);
        bench_gemv(64, 256);
        bench_gemv(128, 1024);
        bench_gemv(256, 1024);
        bench_gemv(1024, 256);
        bench_gemv(1024, 1024);
    }

    if filter == "*" || filter == "gemm" {
        bench_gemm(16, 16, 62);
        bench_gemm(16, 32, 32);
        bench_gemm(16, 64, 64);
        bench_gemm(16, 128, 128);
        bench_gemm(32, 128, 128);
        bench_gemm(64, 256, 128);
        bench_gemm(128, 1024, 256);
        bench_gemm(256, 1024, 256);
        bench_gemm(1024, 256, 512);
        bench_gemm(768, 768, 768);
        bench_gemm(1024, 1024, 1024);
        bench_gemm(2048, 2048, 2048);

        // Specific numbers for DLL
        bench_gemm(100, 500, 768); // Forward
        bench_gemm(100, 500, 500); // Forward
        bench_gemm(100, 10, 500); // Forward

        bench_gemm(100, 500, 500); // Backward
        bench_gemm(100, 768, 500); // Backward
    }

    if filter == "*" || filter == "gemm_outer" {
        bench_gemm_outer(16, 128, 128);
        bench_gemm_outer(32, 128, 128);
        bench_gemm_outer(64, 256, 128);
        bench_gemm_outer(128, 1024, 256);
        bench_gemm_outer(256, 1024, 256);
        bench_gemm_outer(1024, 256, 512);
        bench_gemm_outer(768, 768, 768);
    }

    if filter == "*" || filter == "gemm_inner" {
        bench_gemm_inner(16, 128, 128);
        bench_gemm_inner(32, 128, 128);
        bench_gemm_inner(64, 256, 128);
        bench_gemm_inner(128, 1024, 256);
        bench_gemm_inner(256, 1024, 256);
        bench_gemm_inner(1024, 256, 512);
        bench_gemm_inner(768, 768, 768);
    }

    if filter == "*" || filter == "gemm_chain" {
        bench_gemm_chain(16);
        bench_gemm_chain(32);
        bench_gemm_chain(64);
        bench_gemm_chain(128);
        bench_gemm_chain(256);
        bench_gemm_chain(512);
        bench_gemm_chain(768);
    }

    if filter == "*" || filter == "batch_outer" {
        bench_batch_outer(16, 128, 128);
        bench_batch_outer(32, 128, 128);
        bench_batch_outer(64, 256, 128);
        bench_batch_outer(128, 1024, 256);
        bench_batch_outer(256, 1024, 256);
        bench_batch_outer(1024, 256, 512);
        bench_batch_outer(768, 768, 768);

        // Specific numbers for DLL
        bench_batch_outer(500, 10, 100);
        bench_batch_outer(500, 500, 100);
        bench_batch_outer(768, 500, 100);
    }

    if filter == "*" || filter == "bias_add" {
        bench_bias_add(16, 128);
        bench_bias_add(32, 128);
        bench_bias_add(64, 256);
        bench_bias_add(128, 1024);
        bench_bias_add(256, 1024);
        bench_bias_add(1024, 256);
        bench_bias_add(768, 768);
    }
}
