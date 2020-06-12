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

	@Override
	public void settings() {
		size(280, 280);
	}

	@Override
	public void setup() {
		surface.setVisible(false);
		loadImagesFromFile(System.getProperty("user.dir") + "/src/data/numberID/train/",
				"train-images-idx3-ubyte-uncompressed", "C:\\oroarmor\\numberID\\train\\", 60000);
		System.out.println("Training done, switching to test");
		loadImagesFromFile(System.getProperty("user.dir") + "/src/data/numberID/test/",
				"t10k-images-idx3-ubyte-uncompressed", "C:\\oroarmor\\numberID\\test\\", 10000);
	}

	public void loadImagesFromFile(String filePath, String fileName, String savePath, int imageAmount) {
		FileInputStream fos = null;
		File imageFile = new File(filePath + fileName);
		byte[] imageByte = new byte[imageAmount * 28 * 28 + 16];

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

		for (int i = 0; i < imageAmount; i++) {
			int[] pixels = new int[28 * 28];

			for (int j = 0; j < pixels.length; j++) {
				byte value = imageByte[j + 16 + 28 * 28 * i];
				pixels[j] = color(value >= 0 ? value : value + 255);
			}

			PImage testI = new PImage(28, 28);
			testI.loadPixels();
			testI.pixels = pixels;
			testI.updatePixels();

//			testI.resize(280, 280);

			testI.save(savePath + "images/" + i + ".png");
			if (i % 1000 == 0) {
				System.out.println((float) i / (float) imageAmount / 100);
			}

		}
	}

	@Override
	public void draw() {
//		image(firstNum, 0,0,280,280);
	}

}
