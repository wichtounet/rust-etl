# rust-etl

A very simple Expression Templates Library (ETL) in Rust. 

The goal of this project is (currently) for me to learn Rust. And I want to see the gaps between Rust and C++.

## Issues

Here are some of the issues encountered in this library.

### Generics

The main issue I am facing is the generics support in Rust. The Rust compiler does not know which type it is currently processing in the generic code. This means that we have to rely entirely on traits to ensure each operation can be called. This makes the code much more complicated, especially in a library like some expression templates. The advantage is that the code compiles very quickly.

### Mixing mut and immutable

The Rust borrow checker is very powerful. And it is a great addition to a language since it prevents many issues. However, it is also very strict. The main issue comes from trying to mix immutable and mutable version of something. In an expression `x = b * x + z`, we simply cannot write at once because we would either mix one mutable and one immutable reference to `x` or we would have two mutable references to `x` which is forbidden.


### SIMD and Generics

Rust has multiple SIMD library, including `std::simd` in the standard library itself (curently experimental). This works well and is very complete. But there is *one major issue*: It does not work well at all with generic code. I have been trying to get a generic kernel optimized with SIMD  but without much success. Some of the things work relatively well, but many operations are gated behind `SimdFloat` and `SimdInt`, like `reduce_sum`. As a result, the code can only be generic for integers or float, not both, which is highly inconvenient in a library like this.
