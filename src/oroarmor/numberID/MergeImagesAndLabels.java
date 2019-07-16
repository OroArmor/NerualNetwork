package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

import oroarmor.neuralnetwork.matrix.Matrix;
import processing.core.PApplet;
import processing.core.PImage;

public class MergeImagesAndLabels extends PApplet {

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.MergeImagesAndLabels");
	}

	public void settings() {
		size(280, 280);
	}

	public void setup() {
//		surface.setVisible(false);
		mergeImagesAndLabels("C:\\oroarmor\\numberID\\train\\labels.txt", "C:\\oroarmor\\numberID\\train\\images\\",
				"TEMP", 60000);
	}

	public Map<Matrix, Character> mergeImagesAndLabels(String textFilePath, String imagePath, String savePath,
			int imageAmount) {
		Map<Matrix, Character> map = new HashMap<Matrix, Character>();
		FileInputStream textFile = null;
		try {
			textFile = new FileInputStream(new File(textFilePath));
			for (int i = 1216; i < 1217; i++) {
				Matrix imageData = getImageData(imagePath + i + ".png");
				Character value = null;
				value = (char) textFile.read();
				map.put(imageData, value);
				System.out.println(i);
			}
			textFile.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return map;
	}

	private Matrix getImageData(String string) {
//		System.out.println(string);
		PImage image = loadImage(string);

		image(image, 0, 0, 280, 280);

		image.loadPixels();

		double[] matrixArray = new double[image.height * image.width];

		for (int i = 0; i < matrixArray.length; i++) {
			matrixArray[i] = (double) brightness(image.pixels[i]) / 255f;
		}

		return new Matrix(matrixArray, matrixArray.length, 1);
	}

	public void draw() {
	}

}
