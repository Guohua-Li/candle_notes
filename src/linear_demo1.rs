use candle_core::{
    Device::Cpu,
    Result,
    Tensor
};

const FEATURES: usize = 4;
const UNITS1: usize = 6;
const UNITS2: usize = 2;

struct Model {
    weight1: Tensor,
    weight2: Tensor,
}

impl Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.matmul(&self.weight1)?;
        x.matmul(&self.weight2)
    }
}

fn main() -> Result<()> {
    let w1 = Tensor::randn(0f32, 1.0, (FEATURES, UNITS1), &Cpu)?;
    let w2 = Tensor::randn(0f32, 1.0, (UNITS1, UNITS2 ), &Cpu)?;

    let model = Model { weight1: w1, weight2: w2 };
    let dummy_input = Tensor::randn(0f32, 1.0, (2, FEATURES), &Cpu)?;

    let output = model.forward(&dummy_input)?;
    println!("output:\n{}", output);
    Ok(())
}
