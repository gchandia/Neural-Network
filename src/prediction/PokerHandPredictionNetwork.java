package prediction;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import network.NeuralNetwork;
import network.Neuron;
import network.NeuronLayer;

public class PokerHandPredictionNetwork {
  public List<List<String>> readFile(File file) {
    List<List<String>> lines = new ArrayList<>();
    Scanner inputStream;

    try{
        inputStream = new Scanner(file);
        
        //esto permite manipular diez mil lineas de datos
        for (int i = 0; i < 10000; i++) {
            String line= inputStream.next();
            String[] values = line.split(",");
            lines.add(Arrays.asList(values));
        }

        inputStream.close();
    }catch (FileNotFoundException e) {
        e.printStackTrace();
    }
    
    return lines;
  }
  
  public int[][] getDataAsArray(List<List<String>> data) {
    int[][] dataArray = new int[10000][11];
    //primer elemento debe ser parchado o si no arroja error por razon que se desconoce
    dataArray[0][0] = 1;
    
    for (int i = 0; i < data.size(); i++) {
      for (int j = 0; j < data.get(i).size(); j++) {
        if(i + j != 0) {
          dataArray[i][j] = Integer.parseInt(data.get(i).get(j));
        }
      }
    }
    
    return dataArray;
  }
  
  public double[][] generateInputs(int[][] initialData) {
    double[][] inputs = new double[10000][10];
    
    for (int i = 0; i < initialData.length; i++) {
      for (int j = 0; j < 10; j++) {
        inputs[i][j] = 1.0 * initialData[i][j];
      }
    }
    
    return inputs;
  }
  
  public int[] generatePredictions(int[][] initialData) {
    int[] outputs = new int[10000];
    
    for (int i = 0; i < initialData.length; i++) {
      outputs[i] = initialData[i][10];
    }
    
    return outputs;
  }
  
  public double[][] generateDesiredOutputs(int[] initialData) {
    double[][] outputs = new double[10000][10];
    
    for (int i = 0; i < initialData.length; i++) {
      double[] actual = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      actual[initialData[i]] = 1.0;
    }
    
    return outputs;
  }
  
  public double[] generateWeights(int base, int length) {
    double[] weights = new double[length];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = base * i;
    }
    return weights;
  }
  
  public NeuralNetwork generateNeuralNetwork() {
    NeuralNetwork network;
    NeuronLayer firstLayer = new NeuronLayer(), secondLayer = new NeuronLayer(), thirdLayer = new NeuronLayer();
    //NeuronLayer extraLayer = new NeuronLayer();
    
    for (int i = 0; i < 6; i++) {
      Neuron actual = new Neuron(1.0, 1.0);
      actual.setWeights(generateWeights(5 - i, 10));
      firstLayer.addNeuron(actual);
    }
    
    for (int i = 0; i < 8; i++) {
      Neuron actual = new Neuron(1.0, 1.0);
      actual.setWeights(generateWeights(3 - i, 6));
      secondLayer.addNeuron(actual);
    }
    
    /*for (int i = 0; i < 3; i++) {
      Neuron actual = new Neuron(2.0, 1.5);
      actual.setWeights(generateWeights(0, 3));
      extraLayer.addNeuron(actual);
    }*/
    
    for (int i = 0; i < 10; i++) {
      Neuron actual = new Neuron(0.1 * i, 0.1 * i);
      actual.setWeights(generateWeights(10 - i, 8));
      thirdLayer.addNeuron(actual);
    }
    
    network = new NeuralNetwork(firstLayer);
    network.addLayer(secondLayer);
    //network.addLayer(extraLayer);
    network.addLayer(thirdLayer);
    
    return network;
  }
  
  public double[][] normalizeData(double[][] initialData) {
    double[][] normalizedData = new double[initialData.length][initialData[0].length];
    for (int i = 0; i < initialData.length; i++) {
      for (int j = 0; j < initialData[0].length; j++) {
        if (j % 2 == 0) {
          normalizedData[i][j] = (initialData[i][j] - 1) / 3; //datos valen entre 1 y 4
        } else {
          normalizedData[i][j] = (initialData[i][j] - 1) / 12; //datos valen entre 1 y 13
        }
      }
    }
    
    return normalizedData;
  }
  
  public static void writeRate (String filename, int[] errorRates) throws IOException{
    BufferedWriter outputWriter = null;
    outputWriter = new BufferedWriter(new FileWriter(filename));
    for (int i = 0; i < errorRates.length; i++) {
      outputWriter.write(Integer.toString(errorRates[i]));
      outputWriter.newLine();
    }
    outputWriter.flush();  
    outputWriter.close();  
  }
  
  public static void main(String[] args) {
    PokerHandPredictionNetwork myPoker = new PokerHandPredictionNetwork();
    int[] errorRates = new int[2000];
    
    
    File file= new File("data.txt");
    int[][] myData = myPoker.getDataAsArray(myPoker.readFile(file));
    
    double[][] setOfInputs = myPoker.generateInputs(myData);
    int[] myPredictions = myPoker.generatePredictions(myData);
    
    double[][] desiredOutputs = myPoker.generateDesiredOutputs(myPredictions);
    double[][] setOfNormalizedInputs = myPoker.normalizeData(setOfInputs);

    NeuralNetwork myNetwork = myPoker.generateNeuralNetwork();
    
    long startTime = System.currentTimeMillis();
    
    for (int i = 0; i < 2000; i++) {
      int errorRate = 0;
      myNetwork.train(setOfNormalizedInputs[i], desiredOutputs[i]);
      for (int j = 5000; j < 10000; j++) {
        if (myNetwork.predict(setOfNormalizedInputs[j]) != myPredictions[j]) {
          errorRate++;
        }
      }
      errorRates[i] = errorRate;
    }
    
    /*for (int i = 2000; i > 0; i--) {
      int errorRate = 0;
      myNetwork.train(setOfNormalizedInputs[i], desiredOutputs[i]);
      for (int j = 2000; j < 10000; j++) {
        if (myNetwork.predict(setOfNormalizedInputs[j]) != myPredictions[j]) {
          errorRate++;
        }
      }
      errorRates[i] = errorRate;
    }*/
    
    long stopTime = System.currentTimeMillis();
    long elapsedTime = stopTime - startTime;
    System.out.println(elapsedTime);
    
    try {
      writeRate("rateData.txt", errorRates);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
