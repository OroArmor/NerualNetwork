package com.oroarmor.neural_network.lib.training;

import com.oroarmor.neural_network.lib.matrix.Matrix;

public abstract class GetData<T extends Matrix<T>> {

	public String[] globalArgs;

	public GetData(String[] globalArgs) {
		this.globalArgs = globalArgs;
	}

	public abstract T getData(String[] args);
}
