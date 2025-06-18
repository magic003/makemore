use std::{char, collections::{HashMap, HashSet}, error::Error};
use candle_core::{Tensor, Device, Var};
use candle_nn::encoding;
use makemore;
use rand::{rngs::StdRng, SeedableRng};

/// This program implements the bigram using one-layer neuron. References:
/// * [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3777s) video from Andrej Karparthy.
/// * Second half of the bigrams [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb).
fn main() -> Result<(), Box<dyn Error>> {
    // read words from the file.
    let words = makemore::read_lines("names.txt").expect("Failed to read lines from file");
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

    // create the training set of bigrams (x, y)
    let mut xs: Vec<u32> = vec![];
    let mut ys: Vec<u32> = vec![];
    words.iter().for_each(|word| {
        let mut prev_index = 0;
        for c in word.chars() {
            xs.push(prev_index);
            let current_index = char_to_index[&c] as u32;
            ys.push(current_index as u32);
            prev_index = current_index;
        }
        // Handle the end of the word
        xs.push(prev_index);
        ys.push(0);
    });
    
    let device = &Device::Cpu;
    let size = xs.len();
    let xs = Tensor::from_vec(xs, size, device)?;
    let ys = Tensor::from_vec(ys, size, device)?;
    println!("Training set size: {}", size);
    println!("xs: {}", xs);
    println!("ys: {}", ys);

    // one-hot encoding of xs
    let num_chars = char_to_index.len();
    let xenc = encoding::one_hot(xs, num_chars, 1.0f32, 0.0f32)?;
    println!("One-hot encoded xs: {}", xenc);

    // optimize the parameters using gradient descent
    const LEARNING_RATE: f64 = 80.0;
    let w = Var::randn(0.0f32, 1.0f32, (num_chars, num_chars), device)?;
    for i in 0..30 {
        // Forward pass
        let logits = xenc.matmul(w.as_tensor())?;
        let counts = logits.exp()?;
        let probs = &counts.broadcast_div(&counts.sum_keepdim(1)?)?;

        let loss = probs.gather(&ys.unsqueeze(1)?, 1)?.log()?.mean_all()?.neg()?;
        println!("Epoch: {}, Loss: {}", i, loss.to_scalar::<f32>()?);

        // Backward pass
        let grad_store = loss.backward()?;
        let grad = grad_store.get(&w).expect("Failed to get gradients for w");
        w.set(&w.sub(&(grad * LEARNING_RATE)?)?)?;
    }

    // sampling names
    let mut rng = StdRng::seed_from_u64(1750132625);
    let mut output: Vec<String> = vec![];
    for _ in 0..5 {
        let mut name = String::new();
        let mut last_index = 0;
        loop {
            let x = Tensor::from_vec(vec![last_index as u32], 1, device)?;
            let xenc = encoding::one_hot(x, num_chars, 1.0f32, 0.0f32)?;
            let logits = xenc.matmul(w.as_tensor())?;
            let counts = logits.exp()?;
            let prob = &counts.broadcast_div(&counts.sum_keepdim(1)?)?
                .squeeze(0)?;
            let next_index = makemore::sampling(&prob, &mut rng)?;
            name.push(index_to_char[&next_index]);
            last_index = next_index;
            if last_index == 0 {
                break;
            }
        }

        output.push(name);
    }
    println!("Generated names: {:?}", output);


    Ok(())
}