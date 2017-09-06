package network;

import java.util.ArrayList;

public class NeuronLayer {
  private ArrayList<Neuron> neurons;
  private boolean isOutputLayer;
  private NeuronLayer previousLayer;
  private NeuronLayer nextLayer;
  
  public NeuronLayer() {
    neurons = new ArrayList<>();
    isOutputLayer = true;
    nextLayer = null;
  }
  
  public ArrayList<Neuron> getNeurons() {
    return neurons;
  }
  
  public void addNeuron(Neuron neuron) {
    neurons.add(neuron);
  }
  
  public void setNextLayer(NeuronLayer layer) {
    nextLayer = layer;
    setIsOutputLayer(layer);
    if (layer != null) {
      layer.setPreviousLayer(this);
    }
  }
  
  public void setPreviousLayer(NeuronLayer layer) {
    previousLayer = layer;
  }
  
  private void setIsOutputLayer(NeuronLayer layer) {
    if (layer != null) {
      isOutputLayer = false;
    } else {
      isOutputLayer = true;
    }
  }
  
  public boolean isOutput() {
    if (isOutputLayer == true) {
      return true;
    }
    return false;
  }

  public double[] feed(double[] inputValues) {
    double[] outputValues = new double[neurons.size()];
    for (int i = 0; i < outputValues.length; i++) {
      neurons.get(i).generateOutput(inputValues);
      outputValues[i] = neurons.get(i).getOutput();
    }
    
    if (isOutputLayer) {
      return outputValues;
    } 
    return nextLayer.feed(outputValues);
  }
  
  private void backwardPropagateError() {
    for (int i = 0; i < neurons.size(); i++) {
      double error = 0.0;
      
      for (Neuron neuron : nextLayer.getNeurons()) {
        error += neuron.getWeights()[i] * neuron.getDelta();
      }
      
      neurons.get(i).adjustDeltaWith(error);
    }
    
    if (previousLayer != null) {
      previousLayer.backwardPropagateError();
    }
  }

  public void backwardPropagateError(double[] expected) {
    for (int i = 0; i < neurons.size(); i++) {
      double error = expected[i] - neurons.get(i).getOutput();
      neurons.get(i).adjustDeltaWith(error);
    }
    
    if (previousLayer != null) {
      previousLayer.backwardPropagateError();
    }
  }

  public void updateWeight(double[] initialInputs) {
    double[] inputs;
    double learningRate = 0.5;
    
    if (previousLayer == null) {
      inputs = initialInputs;
    } else {
      ArrayList<Neuron> previousNeurons = previousLayer.getNeurons();
      inputs = new double[previousNeurons.size()];
      for (int i = 0; i < previousNeurons.size(); i++) {
        inputs[i] = previousNeurons.get(i).getOutput();
      }
    }
    
    for (Neuron neuron : neurons) {
      neuron.adjustWeightWithInput(inputs, learningRate);
      neuron.adjustBiasUsingLearningRate(learningRate);
    }
    
    if (nextLayer != null) {
      nextLayer.updateWeight(initialInputs);
    }
  }
}