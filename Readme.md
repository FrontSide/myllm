# My LLM

This is following the book Build A Large Language Model From Scratch by Sebastian Raschka

## How an LLM works

### Transformers 
Most current LLMs are based on the transformer architecture, which is a deep neural network architecture.
Original transformers consist of an encoder which encodes input tokens into a series of vectors that capture contextual information of the input
and a decoder creating output tokens given these contextual vectors.

The contextualisation in the transformers happens through a self-attention meachanism.
This mechanism gives the LLM selective access to the whole input sequence when generating the output one word at a time.

Transformers consist of many layers of encoders and decoders. The layers of decoders in-between the very first and very last layer
are often referred to as the "hidden layers".

The GPT architecture is a variation of the transformer with essentially just the decoder.

## Building an LLM

Following steps will need to be implemented to build an LLM:

- Load the training data
- Tokenise the text (i.e turn the words into numbers)
- Create embedding vectors (turn each token into an n-dimensional vector)
- Create position-aware input embedding vectors (change the embedding vectors for each occurrence of a word in the input text based on its position)
- Pass the embeddings through a self-attention mechanism with with trainable weights (this will further transform the input vectors to also incorporate information about all other inputs - in a given sequence - relative to any one token.)

### Tokenisation and Embeddings
We transform words into vectors by splitting them into tokens.
One tokeniser called BytePairEncoding was used for ChatGPT, here not just entire words are turned into individual tokens but also sub-words and individual letters.
This way it will be possible to represent each possible word with tokens even if it wasn't in the original training data.
It would not be efficient to only use per-character encoding.

Tokens are represented as multi-dimensional vectors then called token embeddings. GPT-3 uses an embedding size of over 12,000 dimensions.

The embedding vectors are then further modified to also encode the position of any given occurrence of a token in the input text.
This means that the same token will be represented as different vector embeddigs based on the position of each occurrence.

### The attention mechanism
As mentioned above, most LLMs are transformers which contain an attention mechanism.

Prior to attention mechanisms, the hidden layers in a network (such as an RNN) had to pass forward the enire hidden state,
which is a compressed representation of the input sequenced as proccesed by the hidden layers up to that point. This can dilute information
or lead to loss of information.

Attention mechanisms allow for the calculation of relevancy. This is done by representing each token in the input as a weighted sum of all other tokens based on their relevancy to this token.
Attention mechanisms don't skip tokens, but rather assign weights to them.

We call the weighted output for any given token the "context vector".

With self-attention, we say each token in the input sequence is attended to by all other tokens in the sequence which allows for the calculation of this weighted sum.
Whereas With general attention mechanisms, we might compare two sequences with each other.

In this implementaions we are also implementing multi-headed attention, which allows us to run the attention mechanism in parallel 
multiple times with different trainable input weights. 
All context vectors for a token are then combined into one.




