package com.oroarmor.neural_network.util;

/**
 * A integer tuple
 *
 * @author OroArmor
 */
public class Dim3 {
    /**
     * The dimensions
     */
    public int x, y, z;

    /**
     * A dimension with only x
     * @param x
     */
    public Dim3(int x) {
        this(x, 1, 1);
    }

    /**
     * A dimension with x and y
     * @param x
     * @param y
     */
    public Dim3(int x, int y) {
        this(x, y, 1);
    }

    /**
     * A dimension with x, y, and z
     * @param x
     * @param y
     * @param z
     */
    public Dim3(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public String toString() {
        return "Dim3 [x=" + x + ", y=" + y + ", z=" + z + "]";
    }
}
