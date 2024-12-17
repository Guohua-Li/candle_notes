// Adopted from: https://rust.marcoinacio.com/data/candle/

use candle_core::{
    Device, // enum
    Tensor, // struct
    DType,  // enum
    IndexOp,// trait
    Result, // enum
    Module, // trait
    backprop::GradStore, // struct
};

use candle_nn::{
    VarMap,          // struct
    linear::{
        Linear,      // struct
        linear,      // function
    },
    loss::mse,       // function
    optim::{
        AdamW,       // struct
        ParamsAdamW, // struct
        Optimizer,   // trait
    },
    var_builder::{
        VarBuilder,     // struct
        VarBuilderArgs, // struct
        SimpleBackend,  // trait
    },
};

use tqdm::tqdm;

const IN_SIZE: usize = 10;
const N_UNITS: usize = 50;
const LABELS:  usize = 2;

const N_SAMPLES: usize = 120;
const EPOCHS:    usize = 6000;




struct DenseNet {
    ln1: Linear,
    ln2: Linear,
}


impl DenseNet {
    fn new(vb: VarBuilder) -> Result<Self> {
        let ln1: Linear = linear(IN_SIZE, N_UNITS, vb.pp("ln1"))?;
        let ln2: Linear = linear(N_UNITS,  LABELS, vb.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }
}


impl Module for DenseNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.ln1.forward(x)?;
        let x: Tensor = x.relu()?;
        self.ln2.forward(&x)
    }
}


fn dataset(x_size: usize, y_size: usize, n_sam: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let beta: Tensor = Tensor::randn(0f32, 1.0, (x_size, y_size), &dev)?;
    let x   : Tensor = Tensor::randn(0f32, 1.0, (n_sam, x_size), &dev)?;
    let eps : Tensor = Tensor::randn(0f32, 1.0, (n_sam, y_size), &dev)?;
    let y   : Tensor = (x.cos()?.matmul(&beta)? + eps)?;
    return Ok((x,y));
}


fn train(
    nn: &DenseNet,
    opt: &mut AdamW,
    x: &Tensor,
    y: &Tensor,
    split: f32,
    epochs: usize) -> Result<Vec::<f32>>
{
    let (samples, _) = x.dims2()?;
    let n = (split * samples as f32).round() as usize;
    println!("{n} training samples, {} validation samples", samples-n);
    let x_trn = x.i(..n)?;
    let y_trn = y.i(..n)?;
    let x_val = x.i(n..)?;
    let y_val = y.i(n..)?;

    let mut val_losses = Vec::<f32>::with_capacity(epochs);
    for epoch in tqdm(0..epochs) {
        let out = &nn.forward(&x_trn)?;
        let loss = mse(&out, &y_trn)?;
        let gradients = loss.backward()?;
        opt.step(&gradients)?;
        if epoch % (epochs / 40) == 0 || epoch == epochs - 1 {
            let out  = &nn.forward(&x_val)?;
            let loss = mse(&out, &y_val)?.to_scalar()?;
            val_losses.push(loss);
        }
    }
    Ok(val_losses)
}


fn main() -> Result<()> {
    let dev: Device = Device::cuda_if_available(0)?;

    let varmap: VarMap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dnn: DenseNet = DenseNet::new(vb)?;

    let mut opt: AdamW = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW::default(),
    )?;

    let (x, y) = dataset(IN_SIZE, LABELS, N_SAMPLES, &dev)?;
    let mut total_val_losses = Vec::<f32>::with_capacity(EPOCHS);
    let losses = train(&dnn, &mut opt, &x, &y, 0.8, EPOCHS)?;
    total_val_losses.extend(losses);
    println!("Losses on validation set:\n{:?}", total_val_losses);
    Ok(())
}
