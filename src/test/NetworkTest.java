package test;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import network.NeuralNetwork;
import network.NeuronLayer;
import network.Neuron;

public class NetworkTest {
  NeuralNetwork myNetwork;
  NeuronLayer layerOne;
  NeuronLayer layerTwo;
  NeuronLayer layerThree;
  NeuronLayer nullLayer;
  Neuron neuronOne;
  Neuron neuronTwo;
  Neuron neuronThree;
  Neuron neuronFour;
  double[] weights;
  double[] threeWeights;
  double[] fourWeights;
  double[] inputs;
  double[] expectedOutputs;
  double[] outputs;
  
  @Before
  public void setUp() {
    neuronOne = new Neuron(1.0, 1.0);
    neuronTwo = new Neuron(1.5, 1.5);
    neuronThree = new Neuron(0.5, 1.0);
    neuronFour = new Neuron(2.0, 2.0);
    layerOne = new NeuronLayer();
    layerTwo = new NeuronLayer();
    layerThree = new NeuronLayer();
    myNetwork = new NeuralNetwork(layerOne);
    weights = new double[2];
    weights[0] = 2.0; weights[1] = 3.0;
    threeWeights = new double[3];
    threeWeights[0] = 2.0; threeWeights[1] = 2.0; threeWeights[2] = 2.0;
    fourWeights = new double[4];
    fourWeights[0] = 3.0; fourWeights[1] = 3.0; fourWeights[2] = 3.0; fourWeights[3] = 3.0; 
    inputs = new double[2];
    inputs[0] = 1.0; inputs[1] = 0.0;
    expectedOutputs = new double[2];
    expectedOutputs[0] = 0.0; expectedOutputs[1] = 0.05;
  }
  
  @Test
  public void neuronTest() {
    assertEquals(1.0, neuronOne.getDelta(), 0.0);
    assertEquals(1.0, neuronOne.getBias(), 0.0);
    neuronOne.setBias(2.0);
    neuronOne.setDelta(2.0);
    neuronOne.setWeights(weights);
    neuronOne.setOutput(1.0);
    assertEquals(2.0, neuronOne.getDelta(), 0.0);
    assertEquals(2.0, neuronOne.getBias(), 0.0);
    assertEquals(2.0, neuronOne.getWeights()[0], 0.0);
    assertEquals(3.0, neuronOne.getWeights()[1], 0.0);
    assertEquals(1.0, neuronOne.getOutput(), 0.0);
    assertEquals(0.0, neuronOne.transferDerivative(neuronOne.getOutput()), 0.0);
    neuronOne.adjustDeltaWith(1.0);
    neuronOne.adjustWeightWithInput(weights, 0.5);
    neuronOne.adjustBiasUsingLearningRate(0.5);
    assertEquals(2.0, neuronOne.getBias(), 0.0);
    assertEquals(2.0, neuronOne.getWeights()[0], 0.0);
    assertEquals(3.0, neuronOne.getWeights()[1], 0.0);
    assertEquals(0.0, neuronOne.getDelta(), 0.0);
    neuronOne.generateOutput(weights);
    assertEquals(1.0, neuronOne.getOutput(), 0.001);    
  }
  
  @Test
  public void layerTest() {
    neuronOne.setBias(0.0);
    neuronOne.setDelta(1.0);
    neuronOne.setWeights(weights);
    
    layerOne.setNextLayer(layerTwo);
    layerOne.addNeuron(neuronOne);
    layerOne.addNeuron(neuronOne);
    
    layerTwo.addNeuron(neuronOne);
    layerTwo.setNextLayer(nullLayer);
    
    assertEquals(2, layerOne.getNeurons().size());
    assertFalse(layerOne.isOutput());
    assertTrue(layerTwo.isOutput());
  }
  
  @Test
  public void networkTest() {
    neuronOne.setWeights(weights);
    neuronTwo.setWeights(threeWeights);
    neuronThree.setWeights(fourWeights);
    neuronFour.setWeights(fourWeights);
    
    myNetwork.addLayer(nullLayer);
    myNetwork.addLayer(layerTwo);
    myNetwork.addLayer(layerThree);
    
    assertNotNull(myNetwork.getFirstLayer());
    assertNotNull(myNetwork.getLastLayer());
    assertEquals(layerThree, myNetwork.getLastLayer());
    
    layerOne.addNeuron(neuronOne);
    layerOne.addNeuron(neuronOne);
    layerOne.addNeuron(neuronOne);
    layerTwo.addNeuron(neuronTwo);
    layerTwo.addNeuron(neuronTwo);
    layerTwo.addNeuron(neuronTwo);
    layerTwo.addNeuron(neuronTwo);
    layerThree.addNeuron(neuronThree);
    layerThree.addNeuron(neuronFour);
    
    outputs = myNetwork.train(inputs, expectedOutputs);
    assertEquals(1.0, outputs[0], 0.05);
    assertEquals(1, myNetwork.predict(inputs));
  }
  
  @Test
  public void finalNetworkTest() {
    Neuron firstNeuron = new Neuron(0.1);
    double[] example = {0.1, 0.2};
    firstNeuron.setWeights(example);
    firstNeuron.setOutput(0.2);
    NeuronLayer firstLayer = new NeuronLayer();
    firstLayer.addNeuron(firstNeuron);
    
    Neuron secondNeuron = new Neuron(0.1);
    double[] exampleTwo = {0.2, 0.3};
    secondNeuron.setWeights(exampleTwo);
    secondNeuron.setOutput(0.3);
    firstLayer.addNeuron(secondNeuron);
    
    Neuron outputNeuron = new Neuron(0.1);
    double[] exampleThree = {0.3, 0.4};
    outputNeuron.setWeights(exampleThree);
    outputNeuron.setOutput(0.2);
    NeuronLayer outputLayer = new NeuronLayer();
    outputLayer.addNeuron(outputNeuron);
    
    NeuralNetwork network = new NeuralNetwork(firstLayer);
    network.addLayer(outputLayer);
    
    double[] input = {0.9, 0.8};
    double[] desiredOutput = {1.0};
    
    network.train(input, desiredOutput);
    
    assertEquals(0.10315220543572089, firstNeuron.getBias(), 0.0);
    assertEquals(0.5866175789173301, firstNeuron.getOutput(), 0.0);
    assertEquals(0.006304410871441775, firstNeuron.getDelta(), 0.0);
    
    assertEquals(0.10405277787531439, secondNeuron.getBias(), 0.0);
    assertEquals(0.6271477663131956, secondNeuron.getOutput(), 0.0);
    assertEquals(0.008105555750628777, secondNeuron.getDelta(), 0.0);
    
    assertEquals(0.14332974979557217, outputNeuron.getBias(), 0.0);
    assertEquals(0.6287468135085144, outputNeuron.getOutput(), 0.0);
    assertEquals(0.08665949959114436, outputNeuron.getDelta(), 0.0);    
  }
}
