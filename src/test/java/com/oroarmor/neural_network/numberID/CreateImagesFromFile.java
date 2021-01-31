package com.oroarmor.neural_network.numberID;

import java.io.InputStream;

import processing.core.PImage;

public class CreateImagesFromFile {
    public static void main(String[] args) {
        loadImagesFromFile("numberID/train/",
                "train-images-idx3-ubyte-uncompressed", "C:\\oroarmor\\numberID\\train\\", 60000);
        System.out.println("Training done, switching to test");
        loadImagesFromFile("numberID/test/",
                "t10k-images-idx3-ubyte-uncompressed", "C:\\oroarmor\\numberID\\test\\", 10000);
    }

    public static void loadImagesFromFile(String filePath, String fileName, String savePath, int imageAmount) {
        InputStream fos;
        byte[] imageByte = new byte[imageAmount * 28 * 28 + 16];

        try {
            fos = CreateImagesFromFile.class.getClassLoader().getResourceAsStream(filePath + fileName);
            if (fos == null) {
                throw new NullPointerException("Could not find resource");
            }
            int read = fos.read(imageByte);
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

            testI.save(savePath + "images/" + i + ".png");
            if (i % 1000 == 0) {
                System.out.println((float) i / (float) imageAmount * 100);
            }
        }
    }

    public static int color(int gray) {
        if (gray > 255) gray = 255;
        else if (gray < 0) gray = 0;
        return 0xff000000 | (gray << 16) | (gray << 8) | gray;
    }
}
