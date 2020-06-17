package oroarmor.neuralnetwork.network;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import oroarmor.neuralnetwork.matrix.Matrix;

public class NetworkSaver {

	@SuppressWarnings("unchecked")
	public static <T extends Matrix<T>> NeuralNetwork<T> loadNetworkFromFile(String filePath, String fileName) {
		ObjectInputStream oos = null;
		FileInputStream fos = null;

		NeuralNetwork<T> nn = null;

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
		ObjectOutputStream oos = null;
		File networkFile = null;
		FileOutputStream fos = null;

		try {
			networkFile = new File(path);
			networkFile.mkdirs();
			networkFile.createNewFile();
			fos = new FileOutputStream(path + fileName);
			oos = new ObjectOutputStream(fos);
			oos.writeObject(network.convertAllToCPU());
			fos.close();
			oos.close();
		} catch (Exception e) {
			System.err.print(e);
		}
	}

}
