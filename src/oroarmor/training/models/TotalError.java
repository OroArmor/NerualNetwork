package oroarmor.training.models;

import java.util.ArrayList;

import oroarmor.layer.Layer;
import oroarmor.matrix.Matrix;

public class TotalError extends TrainingModel {

	double trainingRate;
	
	public TotalError(double trainingRate) {
		super(trainingRate);
		this.trainingRate = trainingRate;
	}

	public void fixErrors(ArrayList<Layer> layers, Matrix[] layerOutputs, Matrix expectedOutput) {
		
		Matrix outputErrors = layerOutputs[layerOutputs.length - 1].subtractMatrix(expectedOutput).multiply(-1);
		
		outputErrors.print();
		
		for(int i = layers.size()-1; i > -1; i++) {
			Layer currentLayer = layers.get(i);
			
			layerOutputs[i].print();
			
			layerOutputs[i].getDerivative(currentLayer.getMatrixFunction());
			
			Matrix gradient = layerOutputs[i].getDerivative(currentLayer.getMatrixFunction());
			
			gradient.print();
			
			gradient.multiplyMatrix(outputErrors);
			gradient.multiply(trainingRate);
		
			Matrix transposedWeights =  currentLayer.getWeights().transpose();
			Matrix weightDelta = gradient.multiplyMatrix(transposedWeights);
			
			currentLayer.setWeights(currentLayer.getWeights().addMatrix(weightDelta));
			currentLayer.setBias(currentLayer.getBias().addMatrix(gradient));
		
		}		
	}
	
}
