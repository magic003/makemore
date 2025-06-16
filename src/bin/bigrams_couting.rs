use std::{char, collections::{HashMap, HashSet}};

use makemore;

fn main() {
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
    let char_to_index: HashMap<char, u32> = chars.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
    let index_to_char: HashMap<u32, char> = chars.iter().enumerate().map(|(i, &c)| (i as u32, c)).collect();
    println!("Character to index map: {:?}", char_to_index);
    println!("Index to character map: {:?}", index_to_char);
    

}