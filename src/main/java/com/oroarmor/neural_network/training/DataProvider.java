package com.oroarmor.neural_network.training;

import com.oroarmor.neural_network.matrix.Matrix;

/**
 * A class that can get the data for training steps
 * @param <T> The matrix type
 * @author OroArmor
 */
public abstract class DataProvider<T extends Matrix<T>> {
    /**
     * The global args
     */
    public Object[] globalArgs;

    /**
     * Creates a new {@link DataProvider} instance
     * @param globalArgs The global args for this {@link DataProvider}
     */
    public DataProvider(Object[] globalArgs) {
        this.globalArgs = globalArgs;
    }

    /**
     * Gets the data for the given args
     * @param args The args for this getData
     * @return The matrix for the data
     */
    public abstract T getData(Object[] args);
}
