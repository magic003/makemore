use std::fs::File;
use std::io::{self, BufReader, BufRead};
use candle_core::{Tensor, Error};
use rand::Rng;

/// Reads lines from a file and returns them as a vector of strings.
pub fn read_lines(filename: &str) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let lines = reader
        .lines()
        .collect::<Result<Vec<String>, io::Error>>()?;

    Ok(lines)
}

/// Samples an index from a tensor based on its probability distribution.
pub fn sampling(tensor: &Tensor, mut rng: impl Rng) -> Result<usize, Error> {
    let cumsum: Vec<f32> = tensor.cumsum(0)?.to_vec1()?;
    let r = rng.random();
    let index = cumsum
        .iter()
        .position(|&x| x >= r)
        .unwrap_or(cumsum.len() - 1);
    return Ok(index);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_read_lines() {
        let lines = read_lines("names.txt").unwrap();

        assert_eq!(32033, lines.len());
        assert_eq!("emma", lines[0]);
        assert_eq!("olivia", lines[1]);
    }

    #[test]
    fn test_sampling() {
        let tensor = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], 4, &Device::Cpu)
            .expect("Failed to create tensor");

        let mut rng = StdRng::seed_from_u64(1750132625);
        
        // 0.01550889
        let index = sampling(&tensor, &mut rng).expect("Failed to sample index");
        assert_eq!(0, index);

        // 0.7570026
        let index = sampling(&tensor, &mut rng).expect("Failed to sample index");
        assert_eq!(3, index);
        
        // 0.62311584
        let index = sampling(&tensor, &mut rng).expect("Failed to sample index");
        assert_eq!(3, index);
    }
}