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
