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
     * @param <T> The matrix class
     * @return The neural network
     */
    @SuppressWarnings("unchecked")
    public static <T extends Matrix<T>> NeuralNetwork<T> loadNetworkFromFile(String filePath, String fileName) {
        ObjectInputStream oos;
        FileInputStream fos;
        NeuralNetwork<T> nn;

        try {
            fos = new FileInputStream(filePath + fileName);
            oos = new ObjectInputStream(fos);
            nn = (NeuralNetwork<T>) oos.readObject();
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
    public static void saveNetworkToFile(NeuralNetwork<?> network, String filePath, String fileName) {
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
