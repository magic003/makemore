use std::{char, collections::{HashMap, HashSet}, error::Error, iter};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, ops, Init, VarMap};
use makemore;
use rand::{rngs::StdRng, seq::{IteratorRandom, SliceRandom}, SeedableRng};

/// This program implements the bigram using multilayer perceptron. References:
/// * [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) video from Andrej Karparthy.
/// * The bigrams MLP [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb).
fn main() -> Result<(), Box<dyn Error>> {
    // read words from the file.
    let mut words = makemore::read_lines("names.txt").expect("Failed to read lines from file");
    println!("Total names: {}", words.len());
    println!("First few names: {:?}", &words[0..10]);

    // create a set of unique characters from the words
    let chars: HashSet<char> = words.iter().flat_map(|w| w.chars()).collect();
    let mut chars: Vec<char> = chars.into_iter().collect();
    chars.sort();
    chars.insert(0, '.');
    println!("Sorted characters: {:?}", chars);

    // create char to index and index to char maps
    let char_to_index: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let index_to_char: HashMap<usize, char> = chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();
    println!("Character to index map: {:?}", char_to_index);
    println!("Index to character map: {:?}", index_to_char);

    // build the dataset of n-grams
    const BLOCK_SIZE: usize = 3;
    let device = &Device::new_metal(0)?;

    let build_dataset = |words: &[String]| -> candle_core::Result<(Tensor, Tensor)> {
        let mut xs: Vec<Vec<usize>> = vec![];
        let mut ys: Vec<usize> = vec![];
        for word in words {
            let mut context:Vec<usize> = vec![0; BLOCK_SIZE];
            for c in word.chars().chain(iter::once('.')) {
                let current_index = char_to_index[&c];
                xs.push(context.clone());
                ys.push(current_index);
                context.remove(0);
                context.push(current_index);
            }
        };

        let size = xs.len();
        let xs_tensor = Tensor::from_vec(
            xs.into_iter().flatten().map(|e| e as u32).collect(),
            (size, BLOCK_SIZE),
            device)?;

        let ys_tensor = Tensor::from_vec(
            ys.into_iter().map(|e| e as u32).collect(),
            size,
            device)?;
        Ok((xs_tensor, ys_tensor))
    };

    let mut rng = StdRng::seed_from_u64(1750132625);
    words.shuffle(&mut rng);
    let n1 = words.len() * 8 / 10; // 80% for training
    let n2 = words.len() * 9 / 10; // next 10% for dev

    let (x_tr, y_tr) = build_dataset(&words[0..n1])?;
    let (x_dev, y_dev) = build_dataset(&words[n1..n2])?;
    let (x_te, y_te) = build_dataset(&words[n2..])?;
    println!("x_tr: {:?}, y_tr: {:?}", x_tr.shape(), y_tr.shape());
    println!("x_dev: {:?}, y_dev: {:?}", x_dev.shape(), y_dev.shape());
    println!("x_te: {:?}, y_te: {:?}", x_te.shape(), y_te.shape());

    // construct neural network
    let num_chars = char_to_index.len();
    let varmap = VarMap::new();

    // embedding layer
    const EMBEDDING_SIZE: usize = 10;
    let embeddings = varmap.get(
        (num_chars, EMBEDDING_SIZE), 
        "embeddings", 
        Init::Randn { mean: 0.0, stdev: 1.0 }, 
        DType::F32, 
        device)?;

    // hidden layer
    const HIDDEN_SIZE: usize = 200;
    let w1 = varmap.get(
        (BLOCK_SIZE * EMBEDDING_SIZE, HIDDEN_SIZE), 
        "w1", 
        Init::Randn { mean: 0.0, stdev: 1.0 }, 
        DType::F32, 
        device)?;
    let b1 = varmap.get(
        HIDDEN_SIZE, 
        "b1", 
        Init::Randn { mean: 0.0, stdev: 1.0 }, 
        DType::F32, 
        device)?;

    // output layer
    let w2 = varmap.get(
        (HIDDEN_SIZE, num_chars), 
        "w2", 
        Init::Randn { mean: 0.0, stdev: 1.0 }, 
        DType::F32, 
        device)?;
    let b2 = varmap.get(
        num_chars, 
        "b2", 
        Init::Randn { mean: 0.0, stdev: 1.0 }, 
        DType::F32, 
        device)?;

    // optimize the parameters using gradient descent
    println!("Number of parameters: {}", varmap.all_vars().iter().map(|v| v.elem_count()).sum::<usize>());

    let tr_size = x_tr.dim(0)?;
    const BATCH_SIZE: usize = 32;
    for epoch in 0..20000 {
        // get a mini-batch
        let indices: Vec<u32> = (0..(tr_size as u32)).choose_multiple(&mut rng, BATCH_SIZE);
        let x_batch = x_tr.index_select(&Tensor::from_vec(indices.clone(), BATCH_SIZE, device)?, 0)?;
        let y_batch = y_tr.index_select(&Tensor::from_vec(indices.clone(), BATCH_SIZE, device)?, 0)?;
        if epoch == 0 {
            println!("x_batch shape: {:?}", x_batch.shape());
            println!("y_batch shape: {:?}", y_batch.shape());
        }

        // forward pass
        let x_batch_flat = x_batch.flatten_all()?;
        let emb = embeddings.embedding(&x_batch_flat)?.reshape((BATCH_SIZE, BLOCK_SIZE * EMBEDDING_SIZE))?;
        if epoch == 0 {
            println!("emb shape: {:?}", emb.shape());
        }

        let h = (emb.matmul(&w1)?.broadcast_add(&b1))?.tanh()?;
        if epoch == 0 {
            println!("h shape: {:?}", h.shape());
        }

        let logits = (h.matmul(&w2)?.broadcast_add(&b2))?;
        if epoch == 0 {
            println!("logits shape: {:?}", logits.shape());
        }

        let loss = loss::cross_entropy(&logits, &y_batch)?;
        if epoch % 500 == 0 {
            println!("Epoch: {}, Loss: {}", epoch, loss.to_scalar::<f32>()?);
        }

        // backward pass
        let grads = loss.backward()?;
        let learning_rate = if epoch < 5000 {
            0.1
        } else {
            0.01
        };
        for var in varmap.all_vars() {
            if let Some(grad) = grads.get(&var) {
                var.set(&var.sub(&(grad * learning_rate)?)?)?;
            }
        }
    }

    // loss for the training and dev sets
    {
        let emb = embeddings.embedding(&x_tr.flatten_all()?)?.reshape((tr_size, BLOCK_SIZE * EMBEDDING_SIZE))?;
        let h = (emb.matmul(&w1)?.broadcast_add(&b1))?.tanh()?;
        let logits = (h.matmul(&w2)?.broadcast_add(&b2))?;
        let loss = loss::cross_entropy(&logits, &y_tr)?;
        println!("Training set loss: {}", loss.to_scalar::<f32>()?);
    }

    {
        let emb = embeddings.embedding(&x_dev.flatten_all()?)?.reshape((x_dev.dim(0)?, BLOCK_SIZE * EMBEDDING_SIZE))?;
        let h = (emb.matmul(&w1)?.broadcast_add(&b1))?.tanh()?;
        let logits = (h.matmul(&w2)?.broadcast_add(&b2))?;
        let loss = loss::cross_entropy(&logits, &y_dev)?;
        println!("Dev set loss: {}", loss.to_scalar::<f32>()?);
    }

    // sampling names
    let mut output: Vec<String> = vec![];
    for _ in 0..20 {
        let mut name = String::new();
        let mut context:Vec<u32> = vec![0; BLOCK_SIZE];
        loop {
            let x = Tensor::from_vec(context.clone(), BLOCK_SIZE, device)?;
            let emb = embeddings.embedding(&x)?.reshape((1, BLOCK_SIZE * EMBEDDING_SIZE))?;
            let h = (emb.matmul(&w1)?.broadcast_add(&b1))?.tanh()?;
            let logits = (h.matmul(&w2)?.broadcast_add(&b2))?;
            let prob = ops::softmax(&logits, 1)?;

            let next_index = makemore::sampling(&prob.squeeze(0)?, &mut rng)?;
            name.push(index_to_char[&next_index]);

            if next_index == 0 {
                break;
            }
            context.remove(0);
            context.push(next_index as u32);
        }

        output.push(name);
    }
    println!("Generated names: {:?}", output);

    Ok(())
}
