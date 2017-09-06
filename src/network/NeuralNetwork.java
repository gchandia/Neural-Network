package network;

public class NeuralNetwork {
  private NeuronLayer firstLayer;
  private NeuronLayer lastLayer;
  
  
  public NeuralNetwork(NeuronLayer firstLayer) {
    this.setFirstLayer(firstLayer);
    this.setLastLayer(firstLayer);
  }

  public NeuronLayer getFirstLayer() {
    return firstLayer;
  }

  public void setFirstLayer(NeuronLayer firstLayer) {
    this.firstLayer = firstLayer;
  }
  
  public NeuronLayer getLastLayer() {
    return lastLayer;
  }

  public void setLastLayer(NeuronLayer lastLayer) {
    this.lastLayer = lastLayer;
  }
  
  public void addLayer(NeuronLayer layer) {
    lastLayer.setNextLayer(layer);
    if (layer != null) {
      lastLayer = layer;
    }
  }
  
  public double[] feed(double[] inputValues) {
    return firstLayer.feed(inputValues);
  }
  
  public void backwardPropagateError(double[] expected) {
    this.getLastLayer().backwardPropagateError(expected);
  }
  
  public void updateWeight(double[] initialInputs) {
    this.getFirstLayer().updateWeight(initialInputs);
  }
  
  public double[] train(double[] inputs, double[] expectedOutputs) {
    double[] outputs;
    
    outputs = this.feed(inputs);
    this.backwardPropagateError(expectedOutputs);
    this.updateWeight(inputs);
    
    return outputs;
  }
  
  public int predict(double[] inputs) {
    double[] outputs;
    double max;
    int max_index = 0;
    
    outputs = this.feed(inputs);
    max = outputs[0];
    
    for (int i = 0; i < outputs.length; i++) {
      if (outputs[i] > max) {
        max = outputs[i];
        max_index = i;
      }
    }
    
    return max_index;
  }
}