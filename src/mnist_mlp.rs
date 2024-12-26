use std::fs::File;
use std::io::BufReader;

use rand::thread_rng;
use rand::prelude::SliceRandom;
use tqdm::tqdm;

use candle_core::{
    Device,
    DType,
    Result,
    Tensor,
    Module,
    D
};

use candle_nn::{
    Linear,
    linear,
    loss::nll,
    ops::log_softmax,
    optim::SGD,
    VarBuilder,
    VarMap,
    Optimizer,
};

use candle_datasets::vision::mnist;

const DEVICE: Device = Device::Cpu;

const EPOCHS: usize = 50;
const HIDDEN: usize = 128;
const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

const BSIZE: usize = 100;

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = linear(IMAGE_DIM, HIDDEN, vs.pp("ln1"))?;
        let ln2 = linear(HIDDEN,    LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

#[derive(Debug, serde::Deserialize)]
struct MNISTSample {
    image: Vec<f32>,
    label: u32,
}

fn load_json(fname: &str) -> anyhow::Result<(Tensor, Tensor)> {
    let f = File::open(fname)?;
    let r = BufReader::new(f);
    let vec_data: Vec<MNISTSample> = serde_json::from_reader(r)?;

    let mut images: Vec<f32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    for p in &vec_data {
        let f_vec: Vec<f32> = p.image.clone().into_iter().map(|x| (x as f32)/255.).collect();
        images.extend(f_vec);
        labels.push(p.label);
    }
    let samples: usize = vec_data.len();
    let input_size: usize = vec_data[0].image.len();

    let x_tensor = Tensor::from_vec(images, (samples, input_size), &DEVICE)?;
    let y_tensor = Tensor::from_vec(labels, samples, &DEVICE)?;
    Ok((x_tensor, y_tensor))
}


fn load_data() -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dataset = mnist::load()?;
    println!("train-images: {:?}", dataset.train_images.shape()); // [60000, 784]
    println!("train-labels: {:?}", dataset.train_labels.shape()); // [60000]
    println!("test-images: {:?}", dataset.test_images.shape());   // [10000, 784]
    println!("test-labels: {:?}", dataset.test_labels.shape());   // [10000]
    let train_images = dataset.train_images.to_device(&DEVICE)?;
    let test_images  = dataset.test_images.to_device(&DEVICE)?;
    let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&DEVICE)?;
    let test_labels  = dataset.test_labels.to_dtype(DType::U32)?.to_device(&DEVICE)?;
    Ok((train_images, train_labels, test_images, test_labels))
}

pub fn main() -> anyhow::Result<()> {
    //let (train_x, train_y) = load_json("data/mnist_json/mnist_handwritten_train.json")?;
    //let (test_x,  test_y)  = load_json("data/mnist_json/mnist_handwritten_test.json")?;
    let (train_x, train_y, test_x, test_y) = load_data()?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
    let model = Mlp::new(vs.clone())?;

    let mut opt: SGD = SGD::new(varmap.all_vars(), 0.1)?;

    batch_train(&model, &mut opt, &train_x, &train_y, &test_x, &test_y, EPOCHS)?;
    /*for epoch in 0..EPOCHS {
        let logits = model.forward(&train_x)?;
        let log_sm = log_softmax(&logits, D::Minus1)?;
        let train_loss = nll(&log_sm, &train_y)?;
        opt.backward_step(&train_loss)?;

        let val_logits = model.forward(&test_x)?;
        let sum_ok = val_logits.argmax(D::Minus1)? // index of max value
            .eq(&test_y)?                          // check equality
            .to_dtype(DType::F32)?                 // to F32
            .sum_all()?.to_scalar::<f32>()?;       // sum and convert to scalar
        let test_acc = sum_ok / test_y.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            train_loss.to_scalar::<f32>()?,
            100.0 * test_acc
        );
    }*/
    Ok(())
}

fn train(
    model: &Mlp, opt: &mut SGD,
    train_x: &Tensor, train_y: &Tensor,
    test_x: &Tensor, test_y: &Tensor,
    epochs: usize
) -> anyhow::Result<()> {
    for epoch in 0..epochs {
        let logits = model.forward(&train_x)?;
        let log_sm = log_softmax(&logits, D::Minus1)?;
        let train_loss = nll(&log_sm, &train_y)?;
        opt.backward_step(&train_loss)?;

        let val_logits = model.forward(&test_x)?;
        let sum_ok = val_logits.argmax(D::Minus1)?
            .eq(test_y)?
            .to_dtype(DType::F32)?
            .sum_all()?.to_scalar::<f32>()?;
        let test_acc = sum_ok / test_y.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            train_loss.to_scalar::<f32>()?,
            100.0 * test_acc
        );
    }
    Ok(())
}

fn batch_train(
    model: &Mlp, opt: &mut SGD,
    train_x: &Tensor, train_y: &Tensor,
    test_x: &Tensor, test_y: &Tensor,
    epochs: usize
) -> anyhow::Result<()> {
    let n_batches = train_x.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    for epoch in 0 .. EPOCHS {
        println!("epoch: {epoch:4} ");
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in tqdm(batch_idxs.iter()) {
            let inputs = train_x.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let labels = train_y.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward(&inputs)?;// true
            let log_sm = log_softmax(&logits, D::Minus1)?;
            let train_loss = nll(&log_sm, &labels)?;
            opt.backward_step(&train_loss)?;
            sum_loss += train_loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_x)?;// , false
        let sum_ok = test_logits.argmax(D::Minus1)?  //
            .eq(test_y)?.to_dtype(DType::F32)? //
            .sum_all()?.to_scalar::<f32>()?;         //
        let test_acc = sum_ok / test_y.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_acc
        );
    }
    Ok(())
}

/*
pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
}
*/
