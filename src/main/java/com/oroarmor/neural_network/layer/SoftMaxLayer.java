package com.oroarmor.neural_network.layer;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.KeepPositiveFunction;
import com.oroarmor.neural_network.matrix.function.SoftMaxFunction;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * A softmax layer. All values are normalized so that the sum of the matrix is 1 after applying a {@link FeedForwardLayer} step
 * @param <T> The Matrix class
 * @author Eli Orona
 */
public class SoftMaxLayer<T extends Matrix<T>> extends FeedForwardLayer<T> {
    private static final long serialVersionUID = SerializationIndexer.SOFT_MAX_LAYER_ID;

    /**
     * The softmax function for the layer
     */
    protected SoftMaxFunction softMax = new SoftMaxFunction();

    /**
     * Creates a new {@link SoftMaxLayer}
     * @param neurons The number of output neurons
     * @param type The type of the matrix for the layer
     */
    public SoftMaxLayer(int neurons, Matrix.MatrixType type) {
        super(neurons, type);
    }

    @Override
    public T feedForward(T inputs) {
        return super.feedForward(inputs).applyFunction(softMax);
    }
}
