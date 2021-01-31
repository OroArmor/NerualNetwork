package com.oroarmor.neural_network.lib.network;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.oroarmor.neural_network.lib.matrix.Matrix;

public class NetworkSaver {

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

	public static void saveNetworkToFile(NeuralNetwork<?> network, String fileName, String path) {
		ObjectOutputStream oos;
		File networkFile;
		FileOutputStream fos;

		try {
			networkFile = new File(path);
			boolean dirs = networkFile.mkdirs();
			boolean newFile = networkFile.createNewFile();
			fos = new FileOutputStream(path + fileName);
			oos = new ObjectOutputStream(fos);
			oos.writeObject(network.convertAllToCPU());
			fos.close();
			oos.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
