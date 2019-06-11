package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;

import processing.core.PApplet;

public class CreateLabelsFromFile extends PApplet {

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.CreateLabelsFromFile");
	}

	public void settings() {
	}

	public void setup() {
		surface.setVisible(false);
		loadTextFromFile(System.getProperty("user.dir") + "/src/data/numberID/test/",
				"t10k-labels-idx1-ubyte-uncompressed", "C:\\oroarmor\\numberID\\test\\", "labels.txt", 10000);

		loadTextFromFile(System.getProperty("user.dir") + "/src/data/numberID/train/",
				"train-labels-idx1-ubyte-uncompressed", "C:\\oroarmor\\numberID\\test\\", "labels.txt", 60000);
	}

	public void loadTextFromFile(String filePath, String fileName, String savePath, String saveName, int amount) {
		FileInputStream fos = null;
		File imageFile = new File(filePath + fileName);
		byte[] labelBytes = new byte[amount + 8];

		try {
			fos = new FileInputStream(imageFile);
			fos.read(labelBytes);
//			Gunzipper zip = new Gunzipper(imageFile);
//			zip.unzip(new File(filePath+fileName+"-uncompressed"));
//			zip.close();
			fos.close();
		} catch (Exception e) {
			System.out.println(e.getLocalizedMessage());
		}
		try {
			File savefile = new File(savePath + saveName);
			savefile.delete();
			savefile.createNewFile();
			FileOutputStream saveOut = new FileOutputStream(savefile);
			PrintStream stream = new PrintStream(saveOut);
			for (int i = 8; i < labelBytes.length; i++) {
				stream.print(labelBytes[i] + "");
			}

			stream.close();
			saveOut.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void draw() {
	}

}
