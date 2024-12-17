use candle_core::{
    Device,
    Tensor,
    Result,
    DType,
};

fn main() -> Result<()> {
    let device = Device::Cpu;

    let a1: [u32; 3] = [1, 2, 3];              // declares a Rust Array

    let a2: [[u32; 3]; 3] = [[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]];

    let t1 = Tensor::new(&a1, &device)?;
    let t2 = Tensor::new(&a2, &device)?;

    println!("t1:\n{}", t1);
    println!("t2:\n{}", t2);

    let z1 = t1.zeros_like()?; // Tensor::zeros_like(&t1)?
    let z2 = t2.zeros_like()?; // Tensor::zeros_like(&t2)?
    println!("z1:\n{:?}", z1);
    println!("z2:\n{:?}\n", z2);
    println!("t2:\n{:?}\n\n", t2.to_vec2::<u32>()?);

    let z = Tensor::zeros((2, 3), DType::F32, &device)?;
    println!("z: {:?}", z);

    let n = Tensor::randn(0f32, 1.0, (2, 3), &device)?; // values sampled from a normal distribution
    let x = Tensor::rand(0f32, 1.0, (2, 3), &device)?;  // values sampled uniformly between `lo` and `up`
    println!("n: {:?}", n);
    println!("x :\n{}\n\n", x);

    let _randlike_tensor = z.randn_like(0.0, 4.0)?; // Tensor::randn_like(&z, 0.0, 4.0)?

    Ok(())
}
