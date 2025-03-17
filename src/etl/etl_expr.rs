


pub trait EtlValueType : Default + Clone + Copy {}
impl<T: Default + Clone + Copy> EtlValueType for T {}

pub trait EtlExpr<T: EtlValueType> {
    fn size(&self) -> usize;
    fn at(&self, i: usize) -> T;
}

// TODO: It does not seem like I can force Index trait because it must return a reference which
// expressions cannot do
