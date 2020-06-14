package oroarmor.neuralnetwork.util;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.*;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class JCudaKernel {

	public static boolean INIT = false;
	public static CUmodule module = new CUmodule();

	CUfunction function = new CUfunction();
	String name;

	public JCudaKernel(String name) {
		this.name = name;
	}

	public void loadKernel(String kernelPath) {
		checkInit();
		String ptxFileName = JCudaHelper.prepareDefaultCubinFile(kernelPath);
		cuModuleLoad(module, ptxFileName);
		cuModuleGetFunction(function, module, name);
	}

	public void runKernel(Pointer parameters, Dim3 gridSize, Dim3 blockSize) {
		cuLaunchKernel(//
				function, // Kernel Function
				gridSize.x, gridSize.y, gridSize.z, // Grid dimension
				blockSize.x, blockSize.y, blockSize.z, // Block dimension
				0, null, // Shared memory size and stream
				parameters, null // Kernel- and extra parameters
		);

		cuCtxSynchronize();
	}

	public void checkInit() {
		if (!INIT) {
			System.out.println("JCuda was initialized by the program.");
			InitJCuda(true);
		}
	}

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
