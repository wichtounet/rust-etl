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

    pub fn iter(&self) -> std::slice::Iter<'_, i64> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, i64> {
        self.data.iter_mut()
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
