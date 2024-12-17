use candle_core::{
    Device::Cpu,
    Tensor,
    DType,
    Result,
};

fn main() -> Result<()> {
    let a = Tensor::arange(0f32, 6f32,  &Cpu)?.reshape((2, 3))?;
    let b = Tensor::arange(0f32, 12f32, &Cpu)?.reshape((3, 4))?;
    let c = a.matmul(&b)?;  // or c = Tensor::matmul(&a, &b)?;
    println!("a =\n{}\n", a);
    println!("b =\n{}\n", b);
    println!("c:\n{}", c);
    println!("matmul operation doens't support u32 , i64 DTypes.\n");

    let a = Tensor::new(&[[ 0u32,  1],
                          [ 4,     5],
                          [ 8,     9]], &Cpu)?;

    let b = Tensor::new(&[[ 2u32, 3,  6],
                          [ 7,   10, 11]], &Cpu)?;

    let c = a.to_dtype(DType::F32)?.matmul(&b.to_dtype(DType::F32)?)?;
    println!("a =\n{}\n", a);
    println!("b =\n{}\n", b);
    println!("c:\n{}",c);
    Ok(())
}
