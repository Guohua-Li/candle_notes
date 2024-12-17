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


fn main() -> Result<()> {
    let dev: Device = Device::cuda_if_available(0)?;

    let varmap: VarMap = VarMap::new(); // holds named variables. empty
    let vb: VarBuilderArgs<Box<dyn SimpleBackend>> = VarBuilder::from_varmap(
        &varmap,
        DType::F32,
        &dev
    );
    println!("all_vars (before nn creation)\n{:?}", varmap.all_vars());

    let dnn: DenseNet = DenseNet::new(vb)?; // vb: VarBuilderArgs<'_, Box<dyn SimpleBackend>>
    println!("all_vars (after nn creation)\n{:?}", varmap.all_vars());

    let mut opt: AdamW = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr: 1e-1, ..Default::default() } // ParamsAdamW::default()
    )?;

    //for i in varmap.all_vars().iter_mut() {
    //    println!("var:\n{}", i);
    //}

    let (x, y) = dataset(IN_SIZE, LABELS, N_SAMPLES, &dev)?;
    let n = (0.8 * N_SAMPLES as f32).round() as usize;
    let x_train: Tensor = x.i(..n)?;
    let y_train: Tensor = y.i(..n)?;
    let x_valid: Tensor = x.i(n..)?;
    let y_valid: Tensor = y.i(n..)?;
    println!("{n} training samples, {} validation samples", N_SAMPLES-n);

    let mut val_losses = Vec::<f32>::with_capacity(EPOCHS);
    for epoch in tqdm(0..EPOCHS) {
        let out:  Tensor = dnn.forward(&x_train)?;
        let loss: Tensor = mse(&out, &y_train)?;
        let gradients: GradStore = loss.backward()?;
        opt.step(&gradients)?;

        if epoch % (EPOCHS / 40) == 0 || epoch == EPOCHS - 1 {
            let out  = dnn.forward(&x_valid)?;
            let loss = mse(&out, &y_valid)?.to_scalar()?;
            val_losses.push(loss);
        }
    }
    println!("Losses on validation set:\n{:?}", val_losses);
    Ok(())
}


/*
fn print_type_of<T>(_: &T) { 
    println!("{}", std::any::type_name::<T>())
}




A VarMap is a storage container for named variables. Variables can be retrieved from the store, and new variables can be added with initialization configuration if they are missing. VarMap structures can be serialized in the SafeTensors format.

The "VarBuilderArgs" structure is used to retrieve variables, which can be sourced from storage or generated through initialization. The method for retrieving variables is defined in the backend embedded in the `VarBuilder`.

At a high level, a VarBuilder manages the parameters of a model. It can retrieve variables from a pre-trained checkpoint using "VarBuilder::from_mmaped_safetensors" or initialize them for training with "VarBuilder::from_varmap". In this sense, it is similar to how PyTorch automatically tracks parameters. At a lower level, the VarBuilder holds all the Tensors accessed using .get_with_hints. You can navigate ("cd") by changing the prefix of the VarBuilder using .pp.

*/
// https://www.tensorscience.com/posts/a-short-step-by-step-intro-to-machine-learning-in-rust-2024.html


