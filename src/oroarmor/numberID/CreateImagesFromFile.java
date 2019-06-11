package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;
import processing.core.PApplet;
import processing.core.PImage;

public class CreateImagesFromFile extends PApplet {

	PImage firstNum;

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.CreateImagesFromFile");
	}

	public void settings() {
		size(280, 280);
	}

	public void setup() {
//		surface.setVisible(false);
		loadImagesFromFile(System.getProperty("user.dir") + "/src/data/numberID/train/",
				"train-images-idx3-ubyte-uncompressed");
	}

	public void loadImagesFromFile(String filePath, String fileName) {
		FileInputStream fos = null;
		File imageFile = new File(filePath + fileName);
		byte[] imageByte = new byte[60000 * 28 * 28 + 16];

		try {
			fos = new FileInputStream(imageFile);
			fos.read(imageByte);
//			Gunzipper zip = new Gunzipper(imageFile);
//			zip.unzip(new File(filePath+fileName+"-uncompressed"));
//			zip.close();
			fos.close();
		} catch (Exception e) {

			System.out.println(e.getLocalizedMessage());
		}

		for (int i = 0; i < 60000; i++) {
			int[] pixels = new int[28 * 28];

			for (int j = 0; j < pixels.length; j++) {
				byte value = imageByte[j + 16 + 28 * 28 * i];
				pixels[j] = color((int) ((value >= 0) ? value : value + 255));
			}

			PImage testI = new PImage(28, 28);
			testI.loadPixels();
			testI.pixels = pixels;
			testI.updatePixels();

//			testI.resize(280, 280);

			testI.save(filePath+"trainingImages/" + i + ".png");
			if(i%100 == 0)
				System.out.println((float)i/600f);
		}
	}

	public void draw() {
//		image(firstNum, 0,0,280,280);
	}

}
