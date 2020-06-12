package oroarmor.neuralnetwork.training;

import oroarmor.neuralnetwork.matrix.Matrix;

public abstract class GetData<T extends Matrix<T>> {

	public String[] globalArgs;

	public GetData(String[] globalArgs) {
		this.globalArgs = globalArgs;
	}

	public GetData() {

	}

	public abstract T getData(String[] args);
}
