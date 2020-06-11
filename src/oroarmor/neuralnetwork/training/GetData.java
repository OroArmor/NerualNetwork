package oroarmor.neuralnetwork.training;

import oroarmor.neuralnetwork.matrix.Matrix;

public abstract class GetData {

	public String[] globalArgs;

	public GetData(String[] globalArgs) {
		this.globalArgs = globalArgs;
	}

	public GetData() {

	}

	public abstract Matrix getData(String[] args);
}
