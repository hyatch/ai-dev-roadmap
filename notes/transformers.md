# Transformers

- Type of NN architecture that inspires modern LLMs
- Maps tokens to a high-dimensional embedding
- Maps positions in the context window into a high-dimensional embedding
- Predicts on the current token in self-attention:
  - Emitting a query vector and taking the dot product with key vectors of all tokens in the context (vectors are in a semi-high dimensional space)
  - Softmax normalization
  - Dot product with the value vectors
- Self-attention is broken down into multiple heads to learn more complex associations
- Results are passed to a FFNN
- Repeats attention and FFNN block
- Pass into a final linear layer that maps high-dimensional space back into the number of possible tokens to output. 
