% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/neuralfunctions.R
\name{eval_weights}
\alias{eval_weights}
\title{Evaluate Parameters by Setting Neural Network Weights and Evaluating Coil}
\usage{
eval_weights(avec, inputlist, outputs, sel = seq(1, lookforward))
}
\arguments{
\item{avec}{parameter array values}

\item{inputlist}{input list to track where parameters should be assigned.}

\item{outputs}{objective values to evaluate against}

\item{sel}{which values to use in loss function evaluation}
}
\value{
total error
}
\description{
This is a function to pass parameter values into a neural network, which
takes a set of inputs and produces coil paramters. These parameters are
passed into a coil for evaluation against objective outputs.
}
