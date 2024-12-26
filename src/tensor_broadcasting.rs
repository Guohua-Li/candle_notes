

use candle_core::{
    Device::Cpu,
    Tensor,
    Result,
};


fn main() -> Result<()> {
    let a = Tensor::arange(0.0, 8.0, &Cpu)?.reshape((2,4))?; // [2, 4]
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.], &Cpu)?;        // [4]
    //let c = (a + b)?;
    let c = a.broadcast_add(&b)?;
    println!("c:\n{}\n", c);

    let a = Tensor::new(&[[0.0, 10.0, 20.0, 30.0]], &Cpu)?; // [1,4]
    let b = Tensor::new(&[[1.0], [2.0], [3.0]], &Cpu)?;     // [3,1]
    //let c = (a + b)?;
    let _c = a.broadcast_add(&b)?;
    Ok(())
}
