use std::fs::File;
use std::io::{self, BufReader, BufRead};

/// Reads lines from a file and returns them as a vector of strings.
pub fn read_lines(filename: &str) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let lines = reader
        .lines()
        .collect::<Result<Vec<String>, io::Error>>()?;

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_lines() {
        let lines = read_lines("names.txt").unwrap();

        assert_eq!(32033, lines.len());
        assert_eq!("emma", lines[0]);
        assert_eq!("olivia", lines[1]);
    }
}