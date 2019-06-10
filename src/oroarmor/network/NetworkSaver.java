package oroarmor.network;

import java.io.ObjectOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;

public class NetworkSaver {

	public static void saveNetworkToFile(NeuralNetwork network, String fileName, String path) {
		ObjectOutputStream oos = null;
		File networkFile = null;
		FileOutputStream fos = null;

		try {
			networkFile = new File(path);
			networkFile.mkdirs();
			networkFile.createNewFile();
			fos = new FileOutputStream(path+fileName);
			oos = new ObjectOutputStream(fos);
			oos.writeObject(network);
			fos.close();
			oos.close();
		}catch(Exception e) {
			System.err.print(e);
		}
	}
	
	public static NeuralNetwork loadNetworkFromFile(String filePath, String fileName) {
		ObjectInputStream oos = null;
		FileInputStream fos = null;

		NeuralNetwork nn = null;
		
		try {
			fos = new FileInputStream(filePath+fileName);
			oos = new ObjectInputStream(fos);
			
			nn = (NeuralNetwork) oos.readObject();
			
			fos.close();
			oos.close();
		}catch(Exception e) {
			System.err.print(e);
		}
		return nn;
	}

}
