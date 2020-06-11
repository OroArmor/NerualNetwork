package oroarmor.numberID;

import java.io.*;
import java.util.zip.*;

public class Gunzipper {
	private InputStream in;

	public Gunzipper(File f) throws IOException {
		in = new FileInputStream(f);
	}

	public void unzip(File fileTo) throws IOException {
		OutputStream out = new FileOutputStream(fileTo);
		try {
			in = new GZIPInputStream(in);
			byte[] buffer = new byte[65536];
			int noRead;
			while ((noRead = in.read(buffer)) != -1) {
				out.write(buffer, 0, noRead);
			}
		} finally {
			try {
				out.close();
			} catch (Exception e) {
			}
		}
	}

	public void close() {
		try {
			in.close();
		} catch (Exception e) {
		}
	}
}
