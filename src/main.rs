use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Module};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::{Device, Int, Tensor, backend::Backend};
use image::{GenericImageView, ImageBuffer, Rgb};
use std::env;
use std::error::Error;
use std::fs;

/*
Gaurav Sablok
codeprog@icloud.com

Conceptualization:
- RGB based single neural classification method i implemented in Burn
  for classification of the image coming from the diseased and non-diseased
  datasets.
- I thought of this as i have previously implemented in XgBoost and classification
  outperform the neural where we dont have the bounding box detection defined in YOLO, COCO.
- The idea is that we take the image which is clean and read the RGB coordinates
  and convert them into a tensor
- The same we collect the diseased ones as the tensor and then implement
  the single neural classification model on the same.
- Complete conceptualization to coding four hours offline.

*/

fn main() {
    let args = std::env::args();
    if args.len() < 1usize {
        panic!("argument needed for the machine learning")
    }
    let pathfile = env::args().collect::<Vec<_>>();
    let makevechealthyrun = makevechealthy(&pathfile[1]).unwrap();
    let makevechealthycumrun = makevecdiseasedcummulative(&pathfile[1]).unwrap();
    let makevecdiseased = makevecdiseased(&pathfile[2]).unwrap();
    let makevecdiseasedcumrun = makevechealthycummulative(&pathfile[2]).unwrap();
    let finalhealthyrunclass = normalizevec(
        makevechealthyrun.0,
        makevechealthyrun.1,
        makevecdiseased.0,
        makevecdiseased.1,
    )
    .unwrap();
    let finalhealthycumrun = normalizevec(
        makevechealthycumrun.0,
        makevechealthycumrun.1,
        makevecdiseasedcumrun.0,
        makevecdiseasedcumrun.1,
    )
    .unwrap();
    let _ = healthytrain(finalhealthyrunclass.0, finalhealthyrunclass.1, 100);
    let _ = diseasedtrain(finalhealthycumrun.0, finalhealthycumrun.1, 100);
}

type VECTYPE = (Vec<Vec<u8>>, Vec<usize>);

pub fn makevechealthy(pathfile: &str) -> Result<VECTYPE, Box<dyn Error>> {
    let mut vecimagefinal: Vec<Vec<u8>> = Vec::new();
    let veclabels = arrayfillhealthy(&vecimagefinal);
    for i in fs::read_dir(pathfile.to_string())? {
        let mut vecinter: Vec<Vec<u8>> = Vec::new();
        let openfile = i?.path();
        let path_str = openfile.to_str().unwrap();
        let imgread = image::open(path_str).unwrap().to_rgb8();
        for (_x, _y, pixel) in imgread.enumerate_pixels() {
            let r = pixel.0[0];
            let g = pixel.0[1];
            let b = pixel.0[2];
            let mut vecimage: Vec<u8> = Vec::new();
            vecimage.push(r);
            vecimage.push(g);
            vecimage.push(b);
            vecinter.push(vecimage);
        }
        let finalvecinter = vecinter.into_iter().flatten().collect::<Vec<u8>>();
        vecimagefinal.push(finalvecinter);
    }
    Ok((vecimagefinal, veclabels))
}

/*
Making a collatable tensor for the diseased ones
*/

pub fn makevecdiseased(pathfile: &str) -> Result<VECTYPE, Box<dyn Error>> {
    let mut vecimagefinal: Vec<Vec<u8>> = Vec::new();
    let veclabels = arrayfilldiseased(&vecimagefinal);
    for i in fs::read_dir(pathfile.to_string())? {
        let mut vecinter: Vec<Vec<u8>> = Vec::new();
        let openfile = i?.path();
        let path_str = openfile.to_str().unwrap();
        let imgread = image::open(path_str).unwrap().to_rgb8();
        for (_x, _y, pixel) in imgread.enumerate_pixels() {
            let r = pixel.0[0];
            let g = pixel.0[1];
            let b = pixel.0[2];
            let mut vecimage: Vec<u8> = Vec::new();
            vecimage.push(r);
            vecimage.push(g);
            vecimage.push(b);
            vecinter.push(vecimage);
        }
        let finalvecinter = vecinter.into_iter().flatten().collect::<Vec<u8>>();
        vecimagefinal.push(finalvecinter);
    }
    Ok((vecimagefinal.clone(), veclabels))
}

pub fn makevechealthycummulative(pathfile: &str) -> Result<VECTYPE, Box<dyn Error>> {
    let mut vecimagefinal: Vec<Vec<u8>> = Vec::new();
    let veclabels = arrayfillhealthy(&vecimagefinal);
    for i in fs::read_dir(pathfile.to_string())? {
        let mut vecinter: Vec<Vec<u8>> = Vec::new();
        let openfile = i?.path();
        let path_str = openfile.to_str().unwrap();
        let imgread = image::open(path_str).unwrap().to_rgb8();
        for (_x, _y, pixel) in imgread.enumerate_pixels() {
            let r = pixel.0[0];
            let g = pixel.0[1];
            let b = pixel.0[2];
            let mut vecimage: Vec<u8> = Vec::new();
            vecimage.push(r);
            vecimage.push(g);
            vecimage.push(b);
            vecinter.push(vecimage);
        }
        let finalvecinter = vecinter.into_iter().flatten().collect::<Vec<u8>>();
        vecimagefinal.push(finalvecinter);
    }
    Ok((rgbcost(&vecimagefinal), veclabels))
}

/*
Making a collatable tensor for the diseased ones
*/

pub fn makevecdiseasedcummulative(pathfile: &str) -> Result<VECTYPE, Box<dyn Error>> {
    let mut vecimagefinal: Vec<Vec<u8>> = Vec::new();
    let veclabels = arrayfilldiseased(&vecimagefinal);
    for i in fs::read_dir(pathfile.to_string())? {
        let mut vecinter: Vec<Vec<u8>> = Vec::new();
        let openfile = i?.path();
        let path_str = openfile.to_str().unwrap();
        let imgread = image::open(path_str).unwrap().to_rgb8();
        for (_x, _y, pixel) in imgread.enumerate_pixels() {
            let r = pixel.0[0];
            let g = pixel.0[1];
            let b = pixel.0[2];
            let mut vecimage: Vec<u8> = Vec::new();
            vecimage.push(r);
            vecimage.push(g);
            vecimage.push(b);
            vecinter.push(vecimage);
        }
        let finalvecinter = vecinter.into_iter().flatten().collect::<Vec<u8>>();
        vecimagefinal.push(finalvecinter);
    }
    Ok((rgbcost(vecimagefinal.clone()), veclabels))
}

pub fn arrayfillhealthy(arrayvector: &Vec<Vec<u8>>) -> Vec<usize> {
    let mut arraynew: Vec<usize> = Vec::new();
    let arraylength = arrayvector.len();
    let mut i = 0i32;
    while i < arraylength as i32 {
        arraynew.push(0usize);
        i += 1i32;
    }
    arraynew
}

pub fn arrayfilldiseased(arrayvector: &Vec<Vec<u8>>) -> Vec<usize> {
    let mut arraynew: Vec<usize> = Vec::new();
    let arraylength = arrayvector.len();
    let mut i = 0i32;
    while i < arraylength as i32 {
        arraynew.push(1usize);
        i += 1i32;
    }
    arraynew
}

/*
Total RGB cost for the images
 - What i did is that sum up all the R, G and B values for each pixel
   and return me a tensor fitted vector. In this way, I have a condensed tensor with
   only three coordinates for each vector that can be easily flattened for logistic
   regression.
*/

pub fn rgbcost(arrayvect: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut newvec: Vec<Vec<u8>> = Vec::new();
    for i in 0..arrayvect.len() {
        let mut rvec: Vec<_> = Vec::new();
        let mut gvec: Vec<u8> = Vec::new();
        let mut bvec: Vec<u8> = Vec::new();
        rvec.push(arrayvect[i][0]);
        gvec.push(arrayvect[i][1]);
        bvec.push(arrayvect[i][2]);
        let mut finalvec: Vec<u8> = Vec::new();
        let finalr = rvec.iter().sum();
        let finalg = gvec.iter().sum();
        let finalb = bvec.iter().sum();
        finalvec.push(finalr);
        finalvec.push(finalg);
        finalvec.push(finalb);
        newvec.push(finalvec);
    }
    newvec
}

pub fn healthytrain(
    imagevec: Vec<Vec<u8>>,
    labels: Vec<usize>,
    epochcount: usize,
) -> Result<String, Box<dyn Error>> {
    type MyAutodiffBackend = Autodiff<Cpubackend>;
    #[derive(Module, Debug)]
    pub struct ImageClassify<B: Backend> {
        linear: Linear<B>,
    }
    impl<B: Backend> ImageClassify<B> {
        pub fn neuron(numfeatures: usize, device: &B::Device) -> Self {
            let config = LinearConfig::new(numfeatures, 1).with_bias(true);
            let linear = config.init(device);
            Self { linear }
        }

        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(input)
        }

        pub fn lossestimate(
            &self,
            outputlogits: Tensor<B, 2>,
            target: Tensor<B, 2, Int>,
        ) -> Tensor<B, 0> {
            let loss =
                burn::tensor::loss::binary_cross_entropy_with_logits(outputlogits, target, None);
            loss.mean()
        }
    }
    let device = Device::default();
    let trainingdata = imagevec;
    let label = labels;
    let features = trainingdata[0].len();
    let batchsize = trainingdata.len();
    let flattentensor: Vec<f32> = trainingdata
        .into_iter()
        .flatten()
        .map(|x| x as f32)
        .collect();
    let flattenlabel: Vec<i64> = label.into_iter().map(|x| x as i64).collect();
    let inputtensor = Tensor::<MyAutodiffBackend, 2>::from_floats(flattentensor, &device)
        .reshape([batchsize, features])
        .to_device(device);
    let labeltensor = Tensor::<MyAutodiffBackend, 2, Int>::from_ints(flattenlabel, &device)
        .reshape([batchsize, 1])
        .to_device(device);
    let mut model = ImageClassify::new(features, &device);
    let optimizer = SgdConfig::new().with_learning_rate(0.01).init();
    for i in 0..epochcount.parse::<usize>().unwrap() {
        let outputlogits = model.forward_training(inputtensor.clone());
        let loss = model.lossestimate(outputlogits, labeltensor.clone());
        println!("Loss: {:.4}", loss.clone().into_scalar());
        let grads = loss.backward();
        let gradsparams = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, gradsparams);
        println!("Image classification model trained and loaded");
    }

    Ok("healthy samples trained".to_string())
}

pub fn diseasedtrain(
    imagevec: Vec<Vec<u8>>,
    labels: Vec<usize>,
    epochcount: usize,
) -> Result<String, Box<dyn Error>> {
    type MyAutodiffBackend = Autodiff<Cpubackend>;
    #[derive(Module, Debug)]
    pub struct ImageClassify<B: Backend> {
        linear: Linear<B>,
    }
    impl<B: Backend> ImageClassify<B> {
        pub fn neuron(numfeatures: usize, device: &B::Device) -> Self {
            let config = LinearConfig::new(numfeatures, 1).with_bias(true);
            let linear = config.init(device);
            Self { linear }
        }

        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(input)
        }

        pub fn lossestimate(
            &self,
            outputlogits: Tensor<B, 2>,
            target: Tensor<B, 2, Int>,
        ) -> Tensor<B, 0> {
            let loss =
                burn::tensor::loss::binary_cross_entropy_with_logits(outputlogits, target, None);
            loss.mean()
        }
    }
    let device = Device::default();
    let trainingdata = imagevec;
    let label = labels;
    let features = trainingdata[0].len();
    let batchsize = trainingdata.len();
    let flattentensor: Vec<f32> = trainingdata
        .into_iter()
        .flatten()
        .map(|x| x as f32)
        .collect();
    let flattenlabel: Vec<i64> = label.into_iter().map(|x| x as i64).collect();
    let inputtensor = Tensor::<MyAutodiffBackend, 2>::from_floats(flattentensor, &device)
        .reshape([batchsize, features])
        .to_device(device);
    let labeltensor = Tensor::<MyAutodiffBackend, 2, Int>::from_ints(flattenlabel, &device)
        .reshape([batchsize, 1])
        .to_device(device);
    let mut model = ImageClassify::new(features, &device);
    let optimizer = SgdConfig::new().with_learning_rate(0.01).init();
    for i in 0..epochcount.parse::<usize>().unwrap() {
        let outputlogits = model.forward_training(inputtensor.clone());
        let loss = model.lossestimate(outputlogits, labeltensor.clone());
        println!("Loss: {:.4}", loss.clone().into_scalar());
        let grads = loss.backward();
        let gradsparams = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, gradsparams);
        println!("Image classification model trained and loaded");
    }

    Ok("healthy samples trained on cummulative rgb".to_string())
}

// normalize the combined vector for tensor while maintaining
// the order of the features and labels

pub fn normalizevec(
    a: Vec<Vec<u8>>,
    b: Vec<usize>,
    c: Vec<Vec<u8>>,
    d: Vec<usize>,
) -> Result<(Vec<Vec<u8>>, Vec<usize>), Box<dyn Error>> {
    let mut finala: Vec<Vec<u8>> = Vec::new();
    let mut finalb: Vec<usize> = Vec::new();
    for i in 0..a.len() {
        finala.push(a[i].clone());
    }
    for i in 0..c.len() {
        finala.push(c[i].clone());
    }
    for i in 0..b.len() {
        finalb.push(b[i].clone());
    }
    for i in 0..d.len() {
        finalb.push(d[i].clone());
    }
    Ok((finala, finalb))
}
