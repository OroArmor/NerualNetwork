package oroarmor.neuralnetwork.util;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import jcuda.CudaException;

public class JCudaHelper {

	public static String invokeNvcc(String cuFileName, String targetFileType, boolean forceRebuild,
			String... nvccArguments) {
		if (!"cubin".equalsIgnoreCase(targetFileType) && !"ptx".equalsIgnoreCase(targetFileType)) {
			throw new IllegalArgumentException(
					"Target file type must be \"ptx\" or \"cubin\", but is " + targetFileType);
		}

		int dotIndex = cuFileName.lastIndexOf('.');
		if (dotIndex == -1) {
			dotIndex = cuFileName.length();
		}
		String otuputFileName = cuFileName.substring(0, dotIndex) + "." + targetFileType.toLowerCase();
		File ptxFile = new File(otuputFileName);
		if (ptxFile.exists() && !forceRebuild) {
			return otuputFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new CudaException("Input file not found: " + cuFileName + " (" + cuFile.getAbsolutePath() + ")");
		}
		String modelString = "-m" + System.getProperty("sun.arch.data.model");
		String command = "nvcc ";
		command += modelString + " ";
		command += "-" + targetFileType + " ";
		for (String a : nvccArguments) {
			command += a + " ";
		}
		command += cuFileName + " -o " + otuputFileName;

		try {
			Process process = Runtime.getRuntime().exec(command);

			String errorMessage = new String(toByteArray(process.getErrorStream()));
			int exitValue = 0;
			try {
				exitValue = process.waitFor();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new CudaException("Interrupted while waiting for nvcc output", e);
			}
			if (exitValue != 0) {
				throw new CudaException("Could not create " + targetFileType + " file: " + errorMessage);
			}
		} catch (IOException e) {
			throw new CudaException("Could not create " + targetFileType + " file", e);
		}

		return otuputFileName;
	}

	private static byte[] toByteArray(InputStream inputStream) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte[] buffer = new byte[8192];
		while (true) {
			int read = inputStream.read(buffer);
			if (read == -1) {
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
}