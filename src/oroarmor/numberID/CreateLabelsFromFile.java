package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;
import processing.core.PApplet;
import processing.core.PImage;

public class CreateLabelsFromFile extends PApplet {


	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.CreateLabelsFromFile");
	}

	public void settings() {
		size(280, 280);
	}

	public void setup() {
		loadTextFromFile(System.getProperty("user.dir") + "/src/data/numberID/train/",
				"train-labels-idx1-ubyte");
	}

	public void loadTextFromFile(String filePath, String fileName) {
		FileInputStream fos = null;
		File imageFile = new File(filePath + fileName);
		byte[] imageByte = new byte[60000 * 28 * 28 + 16];

		try {
			fos = new FileInputStream(imageFile);
			fos.read(imageByte);
			Gunzipper zip = new Gunzipper(imageFile);
			zip.unzip(new File(filePath+fileName+"-uncompressed"));
			zip.close();
			fos.close();
		} catch (Exception e) {

			System.out.println(e.getLocalizedMessage());
		}

	}

	public void draw() {
	}

}
