package com.oroarmor.neural_network.lib.layer;

import java.util.Random;

import com.oroarmor.neural_network.lib.matrix.CPUMatrix;
import com.oroarmor.neural_network.lib.matrix.Matrix;
import com.oroarmor.neural_network.lib.matrix.Matrix.MatrixType;
import com.oroarmor.neural_network.lib.matrix.function.KeepPositiveFunction;
import com.oroarmor.neural_network.lib.matrix.function.MatrixFunction;

@SuppressWarnings("unchecked")
public class KeepPositiveLayer<T extends Matrix<T>> extends Layer<T> {

	/**
	 *
	 */
	private static final long serialVersionUID = 11L;
	public T weights;

	public KeepPositiveLayer(int neurons, MatrixType type) {
		super(neurons, type);
	}

	@Override
	public T feedFoward(T inputs) {
		return inputs.applyFunction(new KeepPositiveFunction());
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		// TODO Auto-generated method stub
		return new KeepPositiveFunction();
	}

	@Override
	public int getOutputNeurons() {
		return neurons;
	}

	@Override
	public T[] getParameters() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public T getWeights() {
		// TODO Auto-generated method stub
		return weights;
	}

	@Override
	public void setParameters(T[] parameters) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setup(int inputs) {
		weights = (T) CPUMatrix.randomMatrix(neurons, inputs, new Random(), -1, 1);
	}

	@Override
	public void setWeights(T newWeights) {
		weights = newWeights;
	}

	@Override
	public Layer<CPUMatrix> contvertToCPU() {
		KeepPositiveLayer<CPUMatrix> newLayer = new KeepPositiveLayer<>(neurons, MatrixType.CPU);
		newLayer.weights = weights.toMatrix(MatrixType.CPU);

		return newLayer;
	}

}
