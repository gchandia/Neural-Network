package network;

public class Neuron {
  
  private double[] weights;
  private double bias;
  private double delta;
  private double output;
  
  public Neuron(double bias, double delta) {
    this.bias = bias;
    this.delta = delta;
  }
  
  public double[] getWeights() {
    return weights;
  }
  
  public void setWeights(double[] weights) {
    this.weights = new double[weights.length];
    for (int i = 0; i < weights.length; i++) {
      this.weights[i] = weights[i];
    }
  }
  
  public double getBias() {
    return bias;
  }
  
  public void setBias(double bias) {
    this.bias = bias;
  }
  
  public double getDelta() {
    return delta;
  }
  
  public void setDelta(double delta) {
    this.delta = delta;
  }
  
  public void setOutput(double output) {
    this.output = output;
  }

  public double getOutput() {
    return output;
  }
  
  public double transferDerivative(double output) {
    return output * (1.0 - output);
  }
  
  public void generateOutput(double[] inputs) {
    double total = 0;
    for (int i = 0; i < inputs.length; i++) {
      total += inputs[i] * weights[i];
    }
    this.setOutput(1 / (1 + Math.exp(-(total + bias))));
  }

  public void adjustDeltaWith(double error) {
    delta = error * this.transferDerivative(output);
  }

  public void adjustWeightWithInput(double[] inputs, double learningRate) {
    for (int i = 0; i < inputs.length; i++) {
      weights[i] += learningRate * delta * inputs[i];
    }
  }

  public void adjustBiasUsingLearningRate(double learningRate) {
    bias += learningRate * delta;
  }
}