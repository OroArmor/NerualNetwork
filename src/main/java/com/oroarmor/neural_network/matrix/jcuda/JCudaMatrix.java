package com.oroarmor.neural_network.matrix.jcuda;

import java.util.HashMap;
import java.util.Random;
import java.util.stream.Collectors;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.matrix.jcuda.kernels.functions.AbsKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.functions.RandomKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.AddKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.AddValueKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.MultiplyKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.MultiplyValueKernel;
import com.oroarmor.neural_network.util.SerializationIndexer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import static jcuda.driver.JCudaDriver.*;

/**
 * A Matrix that runs on the GPU
 * @author OroArmor
 */
public class JCudaMatrix implements Matrix<JCudaMatrix> {
    private static final long serialVersionUID = SerializationIndexer.GPU_MATRIX_ID;
    /**
     * A map of the disposers to dispose extra gpu data to save on memory
     */
    private static final HashMap<JCudaMatrix, Runnable> disposers = new HashMap<>();

    // values
    /**
     * The array for the matrix
     */
    protected double[] matrix;
    /**
     * The rows for the matrix
     */
    protected int rows;
    /**
     * The columns for the matrix
     */
    protected int cols;
    /**
     * The pointer to the device matrix
     */
    protected CUdeviceptr deviceMPtr;
    /**
     * The pointers for the properties of the matrix
     */
    protected Pointer matrixPointer, rowPointer, columnPointer, sizePointer;
    /**
     * True if the matrix should not be disposed
     */
    protected boolean keep = false;

    /**
     * True if any changing operation has been run
     */
    protected boolean dirty = true;

    /**
     * Creates an empty matrix (0) with rows and one column
     * @param rows The number of rows
     */
    public JCudaMatrix(int rows) {
        this(rows, 1);
    }

    /**
     * Creates an empty matrix
     * @param rows The number of rows
     * @param cols The number of columns
     */
    public JCudaMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        disposers.entrySet().stream().filter(e -> !e.getKey().keep).collect(Collectors.toList())
                .forEach(e -> disposers.remove(e.getKey()).run());
        createPointersNoBackingArray();
        disposers.put(this, this::dispose);
    }

    /**
     * Creates a new matrix with an array
     * @param matrixArray The array for the matrix
     * @param rows The number of rows
     * @param cols The number of columns
     */
    public JCudaMatrix(double[] matrixArray, int rows, int cols) {
        matrix = matrixArray;
        this.rows = rows;
        this.cols = cols;
        disposers.entrySet().stream().filter(e -> !e.getKey().keep).collect(Collectors.toList())
                .forEach(e -> disposers.remove(e.getKey()).run());
        createPointers();
        disposers.put(this, this::dispose);
    }

    /**
     * Returns a random GPU matrix
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param rand The random number generator for the matrix. Note: this is not used to create a random GPUMatrix
     * @param lowerBound The lower value for the random distribution
     * @param upperBound The upper value for the random distribution
     * @return A random matrix
     */
    public static JCudaMatrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
        JCudaMatrix matrix = new JCudaMatrix(rows, cols);
        matrix.randomize(rand, lowerBound, upperBound);
        return matrix;
    }

    /**
     *
     * @return The device matrix pointer
     */
    public CUdeviceptr getDeviceMPtr() {
        return deviceMPtr;
    }

    /**
     * Creates the pointers with an array
     */
    public void createPointers() {
        int matrixByteSize = rows * cols * Sizeof.DOUBLE;

        deviceMPtr = new CUdeviceptr();
        cuMemAlloc(deviceMPtr, matrixByteSize);
        cuMemcpyHtoD(deviceMPtr, Pointer.to(matrix), matrixByteSize);

        matrixPointer = Pointer.to(deviceMPtr);
        rowPointer = Pointer.to(new int[]{rows});
        columnPointer = Pointer.to(new int[]{cols});
        sizePointer = Pointer.to(new int[]{rows * cols});
    }

    /**
     * Creates the pointers without an array
     */
    public void createPointersNoBackingArray() {
        int matrixByteSize = rows * cols * Sizeof.DOUBLE;

        deviceMPtr = new CUdeviceptr();
        cuMemAlloc(deviceMPtr, matrixByteSize);

        matrixPointer = Pointer.to(deviceMPtr);
        rowPointer = Pointer.to(new int[]{rows});
        columnPointer = Pointer.to(new int[]{cols});
        sizePointer = Pointer.to(new int[]{rows * cols});
    }

    /**
     *
     * @return The pointer to the matrix
     */
    public Pointer getMatrixPointer() {
        return matrixPointer;
    }

    /**
     *
     * @return The pointer for the rows
     */
    public Pointer getRowPointer() {
        return rowPointer;
    }

    /**
     *
     * @return The pointer for the columns
     */
    public Pointer getColumnPointer() {
        return columnPointer;
    }

    /**
     *
     * @return The pointer for the size of the matrix
     */
    public Pointer getSizePointer() {
        return sizePointer;
    }

    /**
     * Copies the data from the GPU to the CPU
     * @return A {@link CPUMatrix} with the same data as this
     */
    public CPUMatrix toCPUMatrix() {
        matrix = new double[rows * cols];
        cuMemcpyDtoH(Pointer.to(matrix), deviceMPtr, rows * cols * Sizeof.DOUBLE);

        return new CPUMatrix(matrix, rows, cols);
    }

    /**
     * Disposes the data on the gpu
     */
    public void dispose() {
        cuMemFree(deviceMPtr);
    }

    /**
     * Make sure the data is not disposed
     * @return this
     */
    public JCudaMatrix keep() {
        keep = true;
        return this;
    }

    @Override
    public JCudaMatrix abs() {
        JCudaMatrix abs = new JCudaMatrix(this.rows, this.cols);
        AbsKernel.getInstance().abs(this, abs);
        dirty = true;
        return abs;
    }

    @Override
    public JCudaMatrix add(double val) {
        JCudaMatrix sum = new JCudaMatrix(this.rows, this.cols);
        AddValueKernel.getInstance().add(this, val, sum);
        dirty = true;
        return sum;
    }

    @Override
    public JCudaMatrix addMatrix(JCudaMatrix other) {
        JCudaMatrix sum = new JCudaMatrix(this.rows, this.cols);
        AddKernel.getInstance().add(this, other, sum);
        dirty = true;
        return sum;
    }

    @Override
    public JCudaMatrix applyFunction(MatrixFunction function) {
        return null;
    }

    @Override
    public JCudaMatrix divide(double val) {
        return multiply(1d / val);
    }

    @Override
    public int getCols() {
        return cols;
    }

    @Override
    public JCudaMatrix getDerivative(MatrixFunction function) {
        return null;
    }

    @Override
    public int getRows() {
        return rows;
    }

    @Override
    public double getValue(int row, int col) {
        if (dirty) {
            dirty = false;
            return this.toCPUMatrix().getValue(row, col);
        }
        return matrix[row * cols + col];
    }

    @Override
    public double[] getValues() {
        if (dirty) {
            dirty = false;
            return this.toCPUMatrix().getValues();
        }
        return matrix;
    }

    @Override
    public double sum() {
        return 0;
    }

    @Override
    public JCudaMatrix hadamard(JCudaMatrix other) {
        return null;
    }

    @Override
    public JCudaMatrix multiply(double val) {
        JCudaMatrix product = new JCudaMatrix(this.rows, this.cols);
        MultiplyValueKernel.getInstance().multiplyValue(this, val, product);
        return product;
    }

    @Override
    public JCudaMatrix multiplyMatrix(JCudaMatrix other) {
        JCudaMatrix product = new JCudaMatrix(rows, other.cols);
        MultiplyKernel.getInstance().multiply(this, other, product);
        return product;
    }

    @Override
    public JCudaMatrix pow(double power) {
        return null;
    }

    @Override
    public void randomize(Random rand, double lowerBound, double upperBound) {
        RandomKernel.getInstance().random(this, rand, lowerBound, upperBound);
    }

    @Override
    public void setValue(int row, int col, double val) {
    }

    @Override
    public JCudaMatrix subtract(double val) {
        return add(-val);
    }

    @Override
    public JCudaMatrix subtractMatrix(JCudaMatrix other) {
        return this.addMatrix(other.multiply(-1));
    }

    @Override
    public JCudaMatrix transpose() {
        return null;
    }

    @Override
    public int getMaxIndex() {
        return 0;
    }
}
