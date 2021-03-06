TIL of O(n log n * r) backprop
===

Tl;dr: O(n log n * r) backprop where n is dominating number of nodes and r is number of layers (and unrolling, if recurrent) exists already, made by the cool kids, and no-one seems to be running around with their hair on fire over this fact, for some reason. I think people should be running around with their hair on fire over this, so I rigged up A. Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) to use it.

Backpropagation is a significant thing because (and only because) it is a big enough speed optimization to enable practical usage of neural nets. Construed as an application of dynamic programming on the deltas, it is easy to see why. Without it, getting the mere numerical gradient of the network per step in non-batch SGD is O(n^(2 * r)), where n is the dominating number of nodes and r is the number of layers in the network (or imagined layers, with BPTT). With it, getting the gradient is O(r * n^2): a very big win. This is why people only use the numerical (non-backpropagation) gradient of multilayer perceptron to check their work, not for any practical purpose: because backpropagation is always better.

I noted while poking about the other day that there is another [big win](http://arxiv.org/pdf/1511.05946v5.pdf) in the mere computational complexity of neural representation from the de Freitas group that is presented in the abstract as a new way to compress neural nets, but also seems to drive down the cost of the gradient-getting to O(r * n log n), called ACDC. Not exact and cool like backpropagation, but a big win nonetheless.

Basically, instead of creating an affine transform by a big matrix multiplication, an affine transform is approximated by layering on top of each other a vector multiplication, a cosine transform, another vector multiplication, and an inverse cosine transform. With convolutional nets, they replicate the CaffeNet's performance pretty well with pretty few layers, interspersed with nonlinearities and permutations. Since the vectors are sized to O(n), the time taken is dominated by the cosine transform, which is done by FFT. (So dynamic programming and divide and conquer at the same time, if you think about it!)

This is very similar to [unitary RNN](http://arxiv.org/pdf/1511.06464v4.pdf) from Bengio group in the complexity win, and in the usage of FFT/DCT and in the permutation, and the notes about unitarity, and basically in the everything. I don't know why one doesn't cite the other, just look at the damned layers.

Remarkably, they have put their [code online](https://github.com/mdenil/acdc-torch), too, so I poked about at creating an absurdly huge recurrent neural net, as they actually suggest. There's a lot of phase transition-like happenings in neural network phenomena where the pseudo-order parameter is the dimension of the activations, not necessarily of the parameters. Moreover, in the empirical poking around that people do, the representational power of the LSTM seems to be much more [extensive](https://arxiv.org/pdf/1602.02410v2.pdf) in nature than non-LSTM RNN's: that is, you can crank up the size a lot, it will help as long as you crank up that size.

I tend to believe that criticality has something to do with it (the argument goes: nonvanishing gradient looks mighty like critical slowing down, so there should be an analogous correlation length divergence to be found somewhere, even though the "lattice" is weird -- hey, look at that extensivity), but I should probably keep my mouth shut until there's some code to back it up. One undoubtedly-premature prediction: you should be able to crank up the size of individual layers of highway net a lot for the same reason. That is, huge but only 3-layer highway nets may work strangely well: definitely going to try this soon.

One idle thought is that it is now actually quite within the realm of feasibility to try for 10^9 nodes, to try to dominate the representation power suggested in [Turing 1950](http://www.loebner.net/Prizef/TuringArticle.html) (bits != activations, of course), and then put it in a seq2seq chat: you just have to get one of those giant-memory cloud boxes and do this whole rigamarole in CPU (the reason why I have not done it is that I have spent all of my $120 AWS monthly budget, so I am waiting for next month), unrolling only a little bit (and somehow this ended up working alright with the Exploring the Limits of Language Modeling paper, although of course theirs is a word LM). Of course, this would overfit a language model by quite a bit even with dropout unless you pulled out the 1B dataset for the occasion.

Installation
===

Fork or just clone the [ACDC repo](https://github.com/mdenil/acdc-torch). Then, you try `luarocks make` and then you realize that doesn't work because they deleted a bunch of requirements that weren't actually required but failed to update the makefile, so you will actually have to checkout commit `f16dfa3c8686` _and then_ `luarocks make`. I couldn't get it to work without a GPU on the box. acdc.ACDC doesn't seem to work with nngraph so if you want to poke around with a recurrent net you'll have to use acdc.FastACDC.

Then, you will be able to fork or clone this repo (after installing torch and all the other requirements, see A. Karpathy's repo for this) and run `run_all.sh` in order to poke at the thing. You can also have the normal LSTM (denseLSTM) (run with `dense_run_all.sh`) and there are a few scripts to compare sizes and the scaling of sizes if you'd like. The BPC performance degradation versus dense network is a bit much with these hyperparameters, but the progression as the network gets bigger seems to basically be alright, and it shouldn't be that hard to degrade less (famous last words). The point at which the asymptotic factors seem to cross between the dense LSTM and the ACDC thing, with the hyperparameters I got, seems to be 8000 - huge in normal LSTM land, but who knows? Recall the [Hochreiter formula](http://www.bioinf.jku.at/publications/older/ch7.pdf) (page 6) for STM in LSTM.

The layered ReLU and permutation dealie that the ACDC folk have in the original paper doesn't seem to work in a recurrent architecture, at least if you just plop it in. A more productive improvement might just be sort of putting a highway-like architecture within the simulated dense layer, but I haven't tried that yet.

Remember when comparing to huge dense LSTM that obnoxious model sizes will lead to obnoxious _saved_ model sizes! CUDA and all this jazz and a big ol' model won't fit in a tiny little 8gb EBS, as I learned to my detriment. Nothing special about the hyperparameters I have set up.

I installed this on two separate boxes and suffice it to say that it will generally be really fiddly. Email me (hleehowon at the big search company's webmail) if you have trouble, you hear? Or if you want to yell at me: this is also acceptable.

License
===

MIT, because Karpathy's char-rnn was MIT
