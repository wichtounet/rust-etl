use std::simd::Simd;

pub trait SimdHelper: Sized + std::ops::Add<Output = Self> {}

impl SimdHelper for Simd<i64, 8> {}
impl SimdHelper for Simd<i32, 8> {}
impl SimdHelper for Simd<f32, 8> {}
impl SimdHelper for Simd<f64, 8> {}

// Rust is pretty much retarded for getting constants out a generic type
pub trait Constants {
    fn one() -> Self;
    fn zero() -> Self;
}

impl Constants for f64 {
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
}

impl Constants for f32 {
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
}

impl Constants for i64 {
    fn one() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
}

impl Constants for i32 {
    fn one() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
}

pub trait Float {
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
}

impl Float for f32 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn ln(self) -> Self {
        self.ln()
    }
}

impl Float for f64 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn ln(self) -> Self {
        self.ln()
    }
}

pub trait Abs {
    fn abs(self) -> Self;
}

impl Abs for i32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for i64 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for f32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for f64 {
    fn abs(self) -> Self {
        self.abs()
    }
}
