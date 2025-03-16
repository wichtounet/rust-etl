

pub struct Vector {
    data: Vec<i64>
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0; size] }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl std::ops::Index<usize> for Vector {
    type Output = i64;

    fn index(&self, index: usize) -> &i64 {
        &self.data[index]
    }

}

impl std::ops::IndexMut<usize> for Vector {

    fn index_mut(&mut self, index: usize) -> &mut i64 {
        &mut self.data[index]
    }

}

fn main() {
    let mut vec: Vector = Vector::new(8);

    for n in 0..vec.size() {
        vec[n] = n as i64;
    }

    for n in 0..vec.size() {
        println!("{}", vec[n]);
    }

    println!("Hello, world!");
}
