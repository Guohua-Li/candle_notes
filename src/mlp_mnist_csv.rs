use std::error::Error;
use rand::thread_rng;
use rand::prelude::SliceRandom;
use tqdm::tqdm;

use candle_core::{
    Device,
    DType,
    Result as Candle_Error,
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
    fn new(vs: VarBuilder) -> Candle_Error<Self> {
        let ln1 = linear(IMAGE_DIM, HIDDEN, vs.pp("ln1"))?;
        let ln2 = linear(HIDDEN,    LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Candle_Error<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

fn fit(
    model: &Mlp,
    opt: &mut SGD,
    train_x: &Tensor, train_y: &Tensor,
    test_x: &Tensor, test_y: &Tensor,
    epochs:usize
) -> anyhow::Result<()> {
    let n_batches = train_x.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    for epoch in 0 .. epochs {
        println!("epoch: {epoch:4} ");
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in tqdm(batch_idxs.iter()) {
            let inputs = train_x.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let labels = train_y.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward(&inputs)?;
            let log_sm = log_softmax(&logits, D::Minus1)?;
            let train_loss = nll(&log_sm, &labels)?;
            opt.backward_step(&train_loss)?;
            sum_loss += train_loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;
        let test_logits = model.forward(&test_x)?;
        let sum_ok = test_logits.argmax(D::Minus1)?
            .eq(test_y)?.to_dtype(DType::F32)?
            .sum_all()?.to_scalar::<f32>()?;
        let test_acc = sum_ok / test_y.dims1()? as f32;
        println!( "{epoch:4} train loss {:8.5} test acc: {:5.2}%", avg_loss, 100. * test_acc );
    }
    Ok(())
}

fn read_mnist_csv(csv_path: &str) -> Result<(Tensor,Tensor), Box<dyn Error>> {

    let r = csv::ReaderBuilder::new().has_headers(true).from_path(csv_path);
    let rdr = match r {
        Ok(rdr) => rdr,
        Err(err) => return Err(Box::new(err)),
    };

    let mut csv_images: Vec<u32> = Vec::new();
    let mut csv_labels: Vec<u32> = Vec::new();

    for r in rdr.into_records() {
        let record = match r {
            Ok(record) => record,
            Err(err)   => return Err(Box::new(err)),
        };

        let row: Vec<u32> = record.iter().map(|d| d.parse().unwrap()).collect();
        csv_images.extend(row[1..].to_vec());
        csv_labels.push(row[0]);
    }
    let size = csv_labels.len();
    let csv_images = csv_images.iter().map(|&x| (x as f32)/255.0).collect::<Vec<_>>();
    let x = Tensor::from_vec(csv_images, (size, IMAGE_DIM), &DEVICE)?;
    let y = Tensor::from_vec(csv_labels, size, &DEVICE)?;

    Ok((x, y))
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let (train_x, train_y) = read_mnist_csv("./data/mnist_csv/mnist_train.csv")?;
    let (test_x,  test_y ) = read_mnist_csv("./data/mnist_csv/mnist_test.csv")?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
    let model = Mlp::new(vs.clone())?;
    let mut opt: SGD = SGD::new(varmap.all_vars(), 0.1)?;
    fit(&model, &mut opt, &train_x, &train_y, &test_x, &test_y, EPOCHS)?;
    Ok(())
}
