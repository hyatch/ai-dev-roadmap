Basically, assume we have a base model for prediction. In this case, we are predicting a name letter by letter using a context window model.

Assume our context window is 3 characters and a batch size of 32. x

The forward pass looks like this:
1. Embedding Table inputs to an embedding dimension (27 characters -> 10 dimensions)
2. First Layer maps from each character embedding to the hidden layer dimensions.
	- (3 characters * 10 dimensions -> 64 dimensions)
	- Use the He Initialization
3. Run it through a BatchNorm
	- Has parameters of gain and bias
	- Makes biases in MLP layers obsolete
4. Output Layer maps from hidden layer back down to 27 characters (alphabet + start/end token)

Mathematical Steps:
1. Access the necessary characters from the embedding table (context window)
	- Concatenate across the columns (from 32x3x10 to 32x30)
2. Run through the layer.
	- $z_1=emb\cdot W1+b1$   becomes (32x30 @ 30x64 -> 32 x 64)
3. Batch Norm.
	- Find the mean across the batches
		- $bnmean = \frac1n*\sum z_1$  (keep the dimensions)
	- Find the difference of each batch from the mean
	- Square the difference
	- Find the variance/standard deviation
		- $\sigma = \frac{1}{(n-1)}*bnd\,^2$   
	- Calculate the inverse
		- $\frac{1}{\sqrt{\sigma+1^{-5}}}$ 
	- Multiply the original difference and the inverse
		- $bnraw$
	- Multiply by gain and add bias (trainable parameters)
		- $\alpha\cdot bnraw+\beta$ 
4. Activation Function
	- Tanh or $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ 
5. Second Layer
	- $h \cdot W2+b2$ from (32x64) to (32 x 27)
6. Loss Function
	- Understand that this is (batch_size x possible_outputs)
	- Cross Entropy Loss
		- Find the maximum likelihood character from the 27 neurons via max
		- Normalize the neurons by subtracting each by the maximum
		- Raise each neuron to e. 
		- Sum across each neuron.
		- Take the inverse of the sum. 
		- Multiply each neuron by this inverse
		- Take the log of each neuron -> (will lead to a negative number for all neruons)
		- Across each batch, find the actual next character and index into our log probabilities. Take the negative log of it (now a positive number). Take the mean across all batches of this negative log prob. 
7. Backpropagate! 
	- loss.backward()

Note that the derivative of the forward pass functions are the same shape. 
#### Well what does loss.backward() look like?
- It keeps a running collection of all the numbers necessary for backpropagation when we conduct forward propagation. 

The Internals of Manual Implementation:
1. Cross Entropy
	- $-logprobs[range(n),Yb].mean() = loss$ 
		- Consider this: Only the probability of the true next neuron contributes to the loss function. All other neurons don't receive a gradient should be zero. 
		- $dlogprobs = \frac1n \cdot true\, neuron$  
	- $probs.log() = logprobs$ 
		- The derivative of log is $1/n$
		- $dprobs = \frac1{probs}*dlogprobs$ (from chain rule)
	- $counts * inverse\,sum\,counts = probs$ 
		- Notice that the inverse sum would be a (32x1) tensor. When multiplied by counts to get probabilities, it was broadcasted to fit the (32x27) shape of counts. 
		- $dinverse = $(counts * dprobs).sum(1)$ over the 27D in each batch  
		- This is because counts is the derivative of the multiplication, dprobs for chain rule, and the summation for the broadcasting. 
		- Results in a (32x1) tensor that matches the shape of inverse sum. 
		- ----------------------------------------------
		- $dcounts = inverse * dprobs$ 
		- Same idea for the derivative of counts, but shapes aren't broadcasted
	- $inverse sum = \frac{1}{sum}$ 
		- Taking the derivative is $-\frac{1}{sum^2}$ then take chain rule
		- $dsum = -\frac{1}{sum^2}\cdot dinverse$ 
	- $counts.sum(1) = sum$
		- This maps a (32 x 27) tensor down to (32x1). We need to replicate $sum$ back up to (32 x 27) by concatenating column wise. 
		- $scaled\, sum = ones.like(counts)*sum$ 
		- Note that we have already computed part of $dcounts$ prior.
		- $dcounts = dcounts + scaled\,sum*dsum$
	- $normalized.exp() = counts$
		- The derivative is simply $counts$ 
		- $dnormalized = counts * dcounts$ 
	- $logits - logitmax = normalized$ 
		- A 32x27 tensor is subtracted by a 32x1 tensor element-wise. 
		- $dlogits = dnormalized$
		- $dlogitmax = -dnormalized.sum(1)$ maps to a 32x1 tensor. 
			- We should note that $logitmax$ are only calculated for stability and have no effect on the overall calculation for loss. This means that $dlogitmax$ will be zero. 
	- $logits.max(1) = logitmax$ 
		- Only the corresponding maximum logit affects the calculation of $logitmax$
		- $dlogits = dlogits + F.one-hot(logit.max(1).indices, 27) * dlogitmax$ 
			- Zeros for all indices except for the logit with the maximum
2. 2nd Linear Layer
	- $h \cdot W_2 + b_2 = logits$ 
	- Recognize that their derivatives must hold the same shape. 
	- $dh$ = $dlogits \cdot W_2^T$   (32 x 27) (27 x 64) -> 32x64
	- $dW_2 = h^T\cdot dlogits$  (64 x 32) (32 x 27) -> 64x27
	- $db_2 = dlogits.sum(0)$  (32x27) -> 1x27
3. Activation
	- $tanh(A) = h$
		- The derivative of $tanh$ is $1- tanh^2(A)$ and $h = tanh(A)$ so:
		- $dA = (1-h^2) * dh$   
4. BatchNorm
	 - $\alpha \cdot bnraw + \beta = A$ 
		 - Note that $bnraw$ and $A$ are shape $(32,64)$; $\alpha$ and $\beta$ are shape $(1,64)$
		 - $d\alpha = (bnraw * dA).sum(0)$
		 - $dbnraw = \alpha * dA$ 
		 - $dbnbias =dA.sum(0)$
	- $bndiff*bn\,inverse\, variance =bnraw$ 
		- Recall that this is part of the batch norm calculation with the fraction.
		- $dbndiff$ = $bn\,inverse\,variance *dbnraw$ 
		- $dbn\,inverse\,variance = (bndiff * dbnraw).sum(0)$ 
	- $\frac{1}{\sqrt{bnvar+1^{-5}}} = bn\,inverse\,variance$
		- $dbnvar = -0.5(bnvar+1^{-5})^{-1.5}*dbn\,inverse\,variance$ 
	- $\frac{1}{n-1}*(bndiff^2).sum(0)=bnvar$
		- There is a collapse of the batch dimension here so we need to scale up.
		- $dbndiff^2 = \frac1{n-1}*torch.oneslike(bndiff^2)*dbnvar$ 
	- $bndiff^2$ = $bndiff^2$ 
		- Obviously.
		- $dbndiff = dbndiff + 2*bndiff*dbndiff$  
	- $PreBN - bnmean = bndiff$ 
		- $dPreBN = dbndiff$
		- $dbnmean = -dbndiff.sum(0)$ scale down to the shape
	- $\frac1n*PreBN.sum(0) =bnmean$ 
		- $dPreBN = dPreBN + \frac1n torch.oneslike(PreBN)*dbnmean$ 
5. 1st Linear Layer
	- $x\cdot W_1+b_1 = PreBN$ 
	- $dx = dPreBN \cdot W_1^T$
	- $dW_1 = x^T \cdot dPreBN$
	- $db_1 = dPreBN.sum(0)$ 
6. Embedding
	- $emb.view(emb[0], -1)=x$ 
		- Revert to original shape
		- $demb = dx.view(emb.shape)$
	- $C[Xb]=emb$ 
		- We fill $dC$ with zeros. 
		- Note that $C$ is of shape $(27,10)$ and $demb$ is of shape $(32,3,10)$ and the input $Xb$ is of shape $(32,3)$. 
		- We iterate through every part of $Xb$ with a nested for loop. 
		- For each character we run into, we update the embedding $dC$ with the corresponding spot inside $demb$. 

Done.

We take the trainable parameters and their gradients:
- $dC, dW_1, db_1, dW_2, db_2, d\alpha, d\beta$ 
Then we apply the update to our parameter's data via
p.data = p.data + learning rate * grad
