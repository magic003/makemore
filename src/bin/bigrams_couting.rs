use std::{char, collections::{HashMap, HashSet}, error::Error};
use candle_core::{Tensor, DType, Device};
use rand::{rngs::StdRng, SeedableRng};
use makemore;

/// This program implements the bigram using couting. References:
/// * [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) video from Andrej Karparthy.
/// * First half of the bigrams [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb).
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
    
    // create a bigram tensor with probabilities
    let size = chars.len();
    let mut bigrams_table: Vec<u32> = vec![0; size * size];
    words.iter().for_each(|word| {
        let mut prev_char = '.';
        for c in word.chars() {
            let prev_index = char_to_index[&prev_char];
            let curr_index = char_to_index[&c];
            bigrams_table[prev_index * size + curr_index] += 1;
            prev_char = c;
        }
        // Handle the end of the word
        let prev_index = char_to_index[&prev_char];
        let end_index = char_to_index[&'.'];
        bigrams_table[prev_index * size + end_index] += 1;
    });
    let bigrams = Tensor::from_vec(bigrams_table, (size, size), &Device::Cpu)?;
    println!("Bigrams by count: {}", bigrams);

    let bigrams_f32 = (&bigrams + 1.0)?
        .to_dtype(DType::F32)?;

    let sums = &bigrams_f32.sum_keepdim(1)?;
    let probs = &bigrams_f32.broadcast_div(&sums)?;
    println!("Bigrams probability: {}", probs);

    // sampling names
    let mut rng = StdRng::seed_from_u64(1750132625);
    let mut output: Vec<String> = vec![];
    for _ in 0..5 {
        let mut name = String::new();
        let mut last_index = 0;
        loop {
            let prob = probs.get(last_index)?;
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

    // negative log likelihood loss
    let mut log_likelihood = Tensor::zeros((), DType::F32, &Device::Cpu)?;
    let mut pairs = 0;
    words.iter().try_for_each(|word| -> Result<(), Box<dyn Error>> {
        let mut prev_index = 0;
        for c in word.chars() {
            let curr_index = char_to_index[&c];
            let logprob = probs.get(prev_index)?
                .get(curr_index)?
                .log()?;
            log_likelihood = (&log_likelihood + logprob)?;
            pairs += 1;
            prev_index = curr_index;
        }
        // Handle the end of the word
        let end_index = 0;
        let logprob = probs.get(prev_index)?
            .get(end_index)?
            .log()?;
        log_likelihood = (&log_likelihood + logprob)?;
        pairs += 1;

        Ok(())
    })?;
    let nll = log_likelihood.neg()?;

    println!("NLL: {}", nll);
    let pairs = Tensor::new(&[pairs as f32], &Device::Cpu)?.reshape(())?;
    println!("NLL mean: {}", (nll / pairs)?);

    Ok(())
}