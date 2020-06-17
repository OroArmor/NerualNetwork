package oroarmor.neuralnetwork.util;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxGetDevice;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuInit;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import jcuda.CudaException;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

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

	public static String prepareDefaultCubinFile(String cuFileName) {
		return invokeNvcc(cuFileName, "cubin", false,
				new String[] { "-dlink", "-arch=sm_" + computeComputeCapability() });
	}

	public static String prepareCubinFile(String cuFileName) {
		return invokeNvcc(cuFileName, "cubin", true,
				new String[] { "-dlink", "-arch=sm_" + computeComputeCapability() });
	}

	public static String prepareDefaultPtxFile(String cuFileName) {
		return invokeNvcc(cuFileName, "ptx", false);
	}

	public static String preparePtxFile(String cuFileName) {
		return invokeNvcc(cuFileName, "ptx", true);
	}

	private static int computeComputeCapability() {
		CUdevice device = new CUdevice();
		int status = cuCtxGetDevice(device);
		if (status != CUresult.CUDA_SUCCESS) {
			throw new CudaException(CUresult.stringFor(status));
		}
		return computeComputeCapability(device);
	}

	private static int computeComputeCapability(CUdevice device) {
		int majorArray[] = { 0 };
		int minorArray[] = { 0 };
		cuDeviceGetAttribute(majorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		cuDeviceGetAttribute(minorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
		int major = majorArray[0];
		int minor = minorArray[0];
		return major * 10 + minor;
	}

	static boolean INIT = false;
	public static CUmodule module = new CUmodule();

	public static void InitJCuda(boolean setExceptions) {
		if (!INIT) {
			JCudaDriver.setExceptionsEnabled(setExceptions);

			JCuda.cudaDeviceReset();

			// Initialize the driver and create a context for the first device.
			cuInit(0);
			CUdevice device = new CUdevice();
			cuDeviceGet(device, 0);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);
			INIT = true;
		}
	}
}