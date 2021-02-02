/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
