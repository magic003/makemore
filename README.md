# makemore
An autoregressive character-level language model for making more things in Rust. It is inspired by the makemore project Andrej Karpathy.
* Youtube video: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo).
* Youtube video: [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I).
* Makemore notebooks on [github](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore). 

I wrote it in Rust for education purpose, and for fun of course. :)

### What

It implements a character-level language model using bigrams. It's implemented in 3 ways:
1. Couting: probability of the next character based on couting.
1. Neuron: probability is learned using a single neuron.
1. MLP: probability is learned using a multilayer perceptron.

### How to run
```sh
% cargo build --release
% ./target/release/bigrams_mlp
```
### License

MIT

