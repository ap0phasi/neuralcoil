# Probabilistic Coils: Nonreal Self-Referencing Bayesian Fields
> This is a set of scripts for generating and experimenting with probabilistic coils

## Table of Contents
* [Installation](#install)
* [General Info](#general-information)
* [Coil Behaviors](#coil-behaviors)
* [Physically-Based Control](#physically-based-control)
* [Pseudo Neural Network Parameterization](#pseudo-neural-network-parameterization)
* [Usage](#usage)
<!-- * [License](#license) -->

##Installation
```
library(devtools)
install_github("ap0phasi/neuralcoil")
```

## General Information
Probabilistic coils are systems of interacting, conserved, nonreal-valued Bayesian fields where the conditionals
are themselves dependent on all state and conditional probabilities, thus making coils
self-referencing. 

Probabilistic coils are inspired by the need for a mathematical framework to describe
dynamic, interconnected, non-hierarchical systems. By using conserved Bayesian fields,
we can describe the flow of discrete state probabilities. By making these self-referencing,
we can describe interdependent probabilistic flows. The generalization into complex and quaternionic
number systems offers wider extensibility. 

## Coil Behaviors 
Probabilistic coils exhibit a number of interesting behaviors. One key behavior is sustained 
aperiodic oscillation. As a result, many coils exhibit chaos. 

It should be emphasized that coils behave deterministically, thus irregular phenomenon is a result of
interconnectedness. 


## Physically-Based Control
Coils can be constructed with a variety of physically-based constraints. For example, locality
can be enforced, preventing the flow of probability to non-neighboring states. Inertial biases can
be imposed, decreasing the flow of probability out of a state. 

Locality constraints can also be used to sever coils, resulting in separate interacting conserved subcoils.
Parameter symmetry can be used to formulate coils with identical parameterizations.


## Neural Network Parameterization
This package demonstrates how neural networks can be used to parameterize coils to generate dynamic fields with desired behavior.


## Usage
Please refer to the vignette for usage and examples. 