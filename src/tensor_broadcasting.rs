

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
    let c = a.broadcast_add(&b)?;
    Ok(())
}



/*
https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html

Broadcasting in coding involves performing operations on arrays of different shapes by stretching smaller arrays to match the dimensions of larger arrays without copying the data. This concept is commonly used in numerical computing libraries like NumPy, enabling element-wise operations on arrays with non-identical shapes for more concise and efficient code.
*/
