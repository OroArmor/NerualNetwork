package com.oroarmor.neural_network.numberID;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.PrintStream;

public class CreateLabelsFromFile {
    public static void main(String[] args) {
        loadTextFromFile("numberID/test/",
                "t10k-labels-idx1-ubyte-uncompressed", "C:\\oroarmor\\numberID\\test\\", "labels.txt", 10000);
        loadTextFromFile("numberID/train/",
                "train-labels-idx1-ubyte-uncompressed", "C:\\oroarmor\\numberID\\train\\", "labels.txt", 60000);
        System.out.println("Done");
    }

    public static void loadTextFromFile(String filePath, String fileName, String savePath, String saveName, int amount) {
        InputStream fos;
        byte[] labelBytes = new byte[amount + 8];

        try {
            fos = CreateLabelsFromFile.class.getClassLoader().getResourceAsStream(filePath + fileName);
            if (fos == null) {
                throw new NullPointerException("Could not find resource");
            }
            int read = fos.read(labelBytes);
            fos.close();
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
        }
        try {
            File f = new File(savePath + saveName);
            boolean deleted = f.delete();
            boolean created = f.createNewFile();
            FileOutputStream saveOut = new FileOutputStream(f);
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
}
