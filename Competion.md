# Competion

Google measure fast and slow model

## TPU

Arbitrary slices of references can be read or updated, subject to implementation constraints. Currently, no restrictions are placed on inputs that are 32-bit wide, but only some slicing patterns are supported for narrower types. Reads and writes that are aligned to multiples of, and have a length that is a multiple of 8 and 128 respectively in the last two dimensions are always supported.  

TPU dimension is recommended to be [1,n,n] as [n,1,1] spam n thread on tpu

## Research paper

[Operator Fusion in XLA: Analysis and Evaluation](https://arxiv.org/pdf/2301.13062.pdf)
Fusion operator in XLA allow TPU to optimize cross layers in a neural network at once isstead of performing optimization on each layer -> reduce memory and increase inference speed as model can do skip jump in network

## Search strategy

● Exhaustive
● Simulated annealing (SA)
● Evolutionary (EVO)
● Model-based optimization (MBO)
● Deep reinforcement learning (RL)
