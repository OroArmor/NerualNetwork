package com.oroarmor.neural_network.layer;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.SoftMaxFunction;

public class SoftMaxLayer<T extends Matrix<T>> extends FeedForwardLayer<T> {
    private static final long serialVersionUID = 13L;

    SoftMaxFunction softMax = new SoftMaxFunction();

    public SoftMaxLayer(int neurons, Matrix.MatrixType type) {
        super(neurons, type);
    }

    @Override
    public T feedForward(T inputs) {
        return super.feedForward(inputs).applyFunction(softMax);
    }

    public SoftMaxFunction getSoftMaxFunction() {
        return softMax;
    }

}
