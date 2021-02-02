package com.oroarmor.neural_network.matrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;

/**
 * An interface that all matrices extend
 * @param <T> The Matrix class
 * @author OroArmor
 */
public interface Matrix<T extends Matrix<T>> extends Serializable {

    /**
     * Returns a random matrix
     * @param type The type of matrix
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param rand The random number generator for the matrix. Note: this is not used to create a random GPUMatrix
     * @param lowerBound The lower value for the random distribution
     * @param upperBound The upper value for the random distribution
     * @param <T> The matrix class, must match the type.
     * @return A random matrix
     */
    @SuppressWarnings("unchecked")
    static <T extends Matrix<T>> T randomMatrix(MatrixType type, int rows, int cols, Random rand,
                                                double lowerBound, double upperBound) {
        switch (type) {
            case CPU:
                return (T) CPUMatrix.randomMatrix(rows, cols, rand, lowerBound, upperBound);
            case JCUDA:
                return (T) JCudaMatrix.randomMatrix(rows, cols, rand, lowerBound, upperBound);
        }
        return null;
    }

    /**
     *
     * @return A new matrix with all values set to their absolute value
     */
    T abs();

    /**
     * Adds {@code val} to all values in the matrix to a new matrix
     * @param val The value to add
     * @return A new matrix with all values increased by {@code val}
     */
    T add(double val);

    /**
     * Adds two matrices into a new matrix
     * @param other The other matrix
     * @throws IllegalArgumentException when the two matrices do not have the same rows and cols
     * @return A new matrix as the sum of the two others
     */
    T addMatrix(T other);

    /**
     * Applies a {@link MatrixFunction} on this matrix and stores into another matrix
     * @param function The function to run
     * @return A new matrix after applying the function
     */
    T applyFunction(MatrixFunction function);

    /**
     * Divides all values by {@code val} in the matrix to a new matrix
     * @param val The value to divide by
     * @return A new matrix with all values divided by {@code val}
     */
    T divide(double val);

    /**
     *
     * @return The number of columns in the matrix
     */
    int getCols();

    /**
     * Gets the derivative of a {@link MatrixFunction} on this matrix and stores into another matrix
     * @param function The function to run
     * @return A new matrix after differentiating the function
     */
    T getDerivative(MatrixFunction function);

    /**
     *
     * @return The number of rows in the matrix
     */
    int getRows();

    /**
     * Gets a value at {@code row, col}
     * @param row The row
     * @param col The column
     * @return The value
     */
    double getValue(int row, int col);

    /**
     *
     * @return All values from the matrix
     */
    double[] getValues();

    /**
     *
     * @return The sum of all values in the matrix
     */
    double sum();

    /**
     * Does element-wise multiplication on this and other
     * @param other The other matrix
     * @return A new matrix with the element-wise product
     */
    T hadamard(T other);

    /**
     * Multiplies all values by {@code val} in the matrix to a new matrix
     * @param val The value to multiply by
     * @return A new matrix with all values multiplied by {@code val}
     */
    T multiply(double val);

    /**
     * Multiplies two matrices into a new matrix
     * @param other The other matrix
     * @throws IllegalArgumentException when the cols for this do not equal the rows for other
     * @return A new matrix as the multiplication of the two others
     */
    T multiplyMatrix(T other);

    /**
     * Raises all values by {@code val} in the matrix to a new matrix
     * @param power The value to raise by
     * @return A new matrix with all values raised by {@code val}
     */
    T pow(double power);

    /**
     * Prints the matrix to the console
     * @param format The decimal format for the matrix
     * @return this
     */
    @SuppressWarnings("unchecked")
    default T print(String format) {
        DecimalFormat df = new DecimalFormat(format);

        for (int i = 0; i < getRows(); i++) {
            System.out.print("| ");
            for (int j = 0; j < getCols(); j++) {
                System.out.print(df.format(getValue(i, j)) + " ");
            }
            System.out.println("|");
        }
        System.out.println(" ");
        return (T) this;
    }

    /**
     * Prints the matrix with a decimal format of {@code "#.##"}
     * @see Matrix#print(String) 
     * @return this
     */
    default T print() {
        return print("#.##");
    }

    /**
     * Randomizes this matrix
     * @param rand The random. Note: not used for GPUMatrix
     * @param lowerBound The lower bound for the random matrix
     * @param upperBound The upper bound for the random matrix
     */
    void randomize(Random rand, double lowerBound, double upperBound);

    /**
     * Sets the value at {@code row, col}
     * @param row The row
     * @param col The column
     * @param val The value
     */
    void setValue(int row, int col, double val);

    /**
     * Subtracts {@code val} to all values in the matrix to a new matrix
     * @param val The value to subtract
     * @return A new matrix with all values decreased by {@code val}
     */
    T subtract(double val);

    /**
     * Subtracts two matrices into a new matrix
     * @param other The other matrix
     * @throws IllegalArgumentException when the two matrices do not have the same rows and cols
     * @return A new matrix as the difference of the two others
     */
    T subtractMatrix(T other);

    /**
     * Transposes the matrix (rows -&gt; cols and cols -&gt; rows)
     * @return The transposed matrix
     */
    T transpose();

    /**
     *
     * @return The maximum index in the matrix
     */
    int getMaxIndex();

    /**
     * Converts this matrix to a different type
     * @param type The type of matrix to transform to
     * @param <S> The matrix class to turn into. Must match type
     * @return A new matrix to the new type
     */
    @SuppressWarnings("unchecked")
    default <S extends Matrix<S>> S toMatrix(MatrixType type) {
        switch (type) {
            case CPU:
                if (this instanceof CPUMatrix)
                    return (S) this;
                if (this instanceof JCudaMatrix)
                    return (S) ((JCudaMatrix) this).toCPUMatrix();
                break;
            case JCUDA:
                if (this instanceof JCudaMatrix)
                    return (S) this;
                if (this instanceof CPUMatrix)
                    return (S) (new JCudaMatrix(this.getValues(), this.getRows(), this.getCols()));
        }
        return null;
    }

    /**
     * The different types of matrices
     */
    enum MatrixType {
        /**
         * Matrices that are stored in RAM and calculations are run on the CPU
         */
        CPU,
        /**
         * Matrices that are stored in GPU RAM and calculation are run on the GPU. NOTE: only works with JCUDA
         */
        JCUDA
    }
}