/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
