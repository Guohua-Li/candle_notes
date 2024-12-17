use candle_core::{Device, Result, Tensor};

const FEATURES: usize = 4;
const UNITS1: usize = 6;
const UNITS2: usize = 2;

struct Layer {
    weight: Tensor,
    bias: Tensor,
}

impl Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    fc1: Layer,
    fc2: Layer,
}

impl Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let weight = Tensor::randn(0f32, 1.0, (FEATURES, UNITS1), &device)?;
    let bias   = Tensor::randn(0f32, 1.0, (UNITS1, ), &device)?;
    let fc1    = Layer { weight, bias };
    let weight = Tensor::randn(0f32, 1.0, (UNITS1, UNITS2), &device)?;
    let bias   = Tensor::randn(0f32, 1.0, (UNITS2, ), &device)?;
    let fc2 = Layer { weight, bias };
    let model = Model { fc1, fc2 };
    let dummy_image = Tensor::randn(0f32, 1.0, (1, FEATURES), &device)?;

    // Inference on the model
    let output = model.forward(&dummy_image)?;
    println!("output:\n {}", output);
    Ok(())
}
