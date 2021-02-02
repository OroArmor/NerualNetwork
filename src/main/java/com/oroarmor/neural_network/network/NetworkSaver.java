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

package com.oroarmor.neural_network.network;

import java.io.*;

import com.oroarmor.neural_network.matrix.Matrix;

/**
 * Saves and reads neural networks
 * @author OroArmor
 */
public class NetworkSaver {
    /**
     * Reads the neural network
     * @param filePath The path to the network file
     * @param fileName The network file
     * @param <N> The {@link AbstractNetwork} class
     * @return The neural network
     */
    @SuppressWarnings("unchecked")
    public static <N extends AbstractNetwork<?>> N loadNetworkFromFile(String filePath, String fileName, Class<N> networkClass) {
        ObjectInputStream oos;
        FileInputStream fos;
        N nn;

        try {
            fos = new FileInputStream(filePath + fileName);
            oos = new ObjectInputStream(fos);
            nn =  (N) oos.readObject();
            fos.close();
            oos.close();
        } catch (Exception e) {
            return null;
        }

        return nn;
    }

    /**
     * Writes the neural network to a file
     * @param network The network
     * @param filePath The file path
     * @param fileName The file name
     */
    public static void saveNetworkToFile(AbstractNetwork<?> network, String filePath, String fileName) {
        ObjectOutputStream oos;
        File networkFile;
        FileOutputStream fos;

        try {
            networkFile = new File(filePath);
            boolean dirs = networkFile.mkdirs();
            boolean newFile = networkFile.createNewFile();
            fos = new FileOutputStream(filePath + fileName);
            oos = new ObjectOutputStream(fos);
            oos.writeObject(network.convertAllToCPU());
            fos.close();
            oos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
