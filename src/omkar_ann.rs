// https://medium.com/@cursor0p/lets-learn-candle-%EF%B8%8F-ml-framework-for-rust-9c3011ca3cd9
// https://github.com/omkar-12bits/candle_blogs/tree/main/ch_1_ANN

use candle_core::{
    DType,
    Device,
    Tensor,
    Result,
};

use candle_nn::{
    linear,
    Linear,
    Module,
    Optimizer,
    VarBuilder,
    VarMap,
    AdamW,
    ParamsAdamW,
    loss::mse,
};

const DEVICE: Device = Device::Cpu;
const EPOCHS: usize = 500;
const HIDDENS: usize = 64;
const OUTPUTS: usize = 1;


#[derive(Debug, serde::Deserialize)]
struct Data {
    #[serde(rename = "X_train")]
    x_train: Vec<Vec<f32>>,
    #[serde(rename = "X_test")]
    x_test:  Vec<Vec<f32>>,
    y_train: Vec<f32>,
    y_test:  Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    let file = std::fs::File::open("fetch_california_housing.json")?; // 20640 rows × 9 columns
    let reader = std::io::BufReader::new(file);
    let data: Data = serde_json::from_reader(reader)?;

    let train_d1 = data.x_train.len();    // 16512
    let train_d2 = data.x_train[0].len(); // 8
    let test_d1 = data.x_test.len();      // 4128
    let test_d2 = data.x_test[0].len();   // 8

    // we can not make tensors from multi dimensional vectors
    let x_train_vec = data.x_train.into_iter().flatten().collect::<Vec<_>>();
    let x_test_vec  = data.x_test.into_iter().flatten().collect::<Vec<_>>();

    // data for training and testing
    let train_x = Tensor::from_vec(x_train_vec, (train_d1, train_d2), &DEVICE)?;
    let test_x  = Tensor::from_vec(x_test_vec,  (test_d1,  test_d2),  &DEVICE)?;
    let train_y = Tensor::from_vec(data.y_train, train_d1, &DEVICE)?;
    let test_y  = Tensor::from_vec(data.y_test,  test_d1,  &DEVICE)?;

    // creating model
    let mut varmap = VarMap::new(); // keeps the track of gradients
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
    let model = SimpleNN::new(train_d2, vb)?;

    println!("all_vars (after nn creation)\n{:?}", varmap.all_vars());
    varmap.load("pytorch_model.safetensors")?;
    println!("all_vars (after loading)\n{:?}", varmap.all_vars());
    let optim_config = ParamsAdamW { lr: 1e-2, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), optim_config)?;

    train_model(&model, &train_x, &train_y, &mut opt, EPOCHS)?;
    evaluate_model(&model, &test_x, &test_y)?;
    Ok(())
}


/*
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
    }
}

pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}
*/


#[derive(Debug)]
struct SimpleNN {
    fc1: Linear,
    fc2: Linear,
}

impl SimpleNN {
    fn new(in_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_dim, HIDDENS, vb.pp("fc1"))?;
        let fc2 = linear(HIDDENS, OUTPUTS, vb.pp("fc2"))?;
        Ok( Self { fc1, fc2 } )
    }
}

impl Module for SimpleNN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(xs)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

fn train_model(model: &SimpleNN, x: &Tensor, y: &Tensor, opt: &mut AdamW, epochs: usize) -> Result<()> {
    for epoch in 0 .. epochs {
        let output = model.forward(x)?;
        let loss = mse(&output.squeeze(1)?, y)?;
        opt.backward_step(&loss)?;        // backward pass and optimization
        if (epoch) % 50 == 0 || epoch == epochs-1 {
            println!("Epoch: {}  Train Loss: {}", epoch, loss.to_scalar::<f32>()?);
        }
    }
    Ok(())
}

fn evaluate_model(model: &SimpleNN, x: &Tensor, y: &Tensor) -> Result<()> {
    let output = model.forward(x)?;
    let loss = mse(&output.squeeze(1)?, y)?;
    println!("Test Loss: {}", loss.to_scalar::<f32>()?);
    Ok(())
}
