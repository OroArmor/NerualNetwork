package com.oroarmor.neural_network.numberID;

import java.io.*;
import java.util.zip.GZIPInputStream;

public class Gunzipper {
    private InputStream in;

    public Gunzipper(File f) throws IOException {
        in = new FileInputStream(f);
    }

    public void unzip(File fileTo) throws IOException {
        try (OutputStream out = new FileOutputStream(fileTo)) {
            in = new GZIPInputStream(in);
            byte[] buffer = new byte[65536];
            int noRead;
            while ((noRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, noRead);
            }
        }
    }
}
