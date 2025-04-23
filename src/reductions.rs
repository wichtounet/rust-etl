use crate::etl_expr::*;

// Reduction Operations

pub fn sum<T: EtlValueType, Expr: EtlExpr<T>>(expr: &Expr) -> T {
    assert!(expr.size() > 0);

    let mut value = T::default();

    for i in 0..expr.size() {
        value += expr.at(i);
    }

    value
}

pub fn asum<T: EtlValueType, Expr: EtlExpr<T>>(expr: &Expr) -> T {
    assert!(expr.size() > 0);

    let mut value = T::default();

    for i in 0..expr.size() {
        let v = expr.at(i);
        if v < T::default() {
            value += -v;
        } else {
            value += v;
        }
    }

    value
}

// TODO: I should find a way to make this better and avoid the cast to u32
// TODO: I should also implement stddev
// TODO: Tests and ensure it works with f32

pub fn mean<T: EtlValueType + From<u32>, Expr: EtlExpr<T>>(expr: &Expr) -> Result<T, &'static str> {
    if expr.size() == 0 {
        return Err("Cannot get mean of empty collection");
    }

    Ok(sum(expr) / From::from(expr.size() as u32))
}

pub fn amean<T: EtlValueType + From<u32>, Expr: EtlExpr<T>>(expr: &Expr) -> Result<T, &'static str> {
    if expr.size() == 0 {
        return Err("Cannot get mean of empty collection");
    }

    Ok(asum(expr) / From::from(expr.size() as u32))
}

pub fn max<T: EtlValueType, Expr: EtlExpr<T>>(expr: &Expr) -> Result<T, &'static str> {
    if expr.size() == 0 {
        return Err("Cannot get max of empty collection");
    }

    let mut max_value = expr.at(0);

    for i in 1..expr.size() {
        if expr.at(i) > max_value {
            max_value = expr.at(i)
        }
    }

    Ok(max_value)
}

pub fn min<T: EtlValueType, Expr: EtlExpr<T>>(expr: &Expr) -> Result<T, &'static str> {
    if expr.size() == 0 {
        return Err("Cannot get min of empty collection");
    }

    let mut min_value = expr.at(0);

    for i in 1..expr.size() {
        if expr.at(i) < min_value {
            min_value = expr.at(i)
        }
    }

    Ok(min_value)
}

// The tests

#[cfg(test)]
mod tests {
    use core::f64;

    use crate::reductions::*;
    use crate::vector::Vector;

    use approx::assert_relative_eq;

    #[test]
    fn basic_sum() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        assert_relative_eq!(sum(&a), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn basic_sum_expr() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        assert_relative_eq!(sum(&(&a + &a)), 10.0, epsilon = 1e-6);
    }

    #[test]
    fn basic_asum() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = -3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        assert_relative_eq!(asum(&a), 15.0, epsilon = 1e-6);
    }

    #[test]
    fn basic_mean() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        match mean(&a) {
            Ok(v) => assert_relative_eq!(v, 1.0, epsilon = 1e-6),
            Err(e) => panic!("Error on mean: {e:?}"),
        }
    }

    #[test]
    fn basic_amean() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        match amean(&a) {
            Ok(v) => assert_relative_eq!(v, 3.0, epsilon = 1e-6),
            Err(e) => panic!("Error on amean: {e:?}"),
        }
    }

    #[test]
    fn basic_max() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        match max(&a) {
            Ok(v) => assert_relative_eq!(v, 4.0, epsilon = 1e-6),
            Err(e) => panic!("Error on max: {e:?}"),
        }
    }

    #[test]
    fn basic_min() {
        let mut a = Vector::<f64>::new(5);

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = -5.0;

        match min(&a) {
            Ok(v) => assert_relative_eq!(v, -5.0, epsilon = 1e-6),
            Err(e) => panic!("Error on min: {e:?}"),
        }
    }
}
