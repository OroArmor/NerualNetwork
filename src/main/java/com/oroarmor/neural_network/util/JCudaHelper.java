package com.oroarmor.neural_network.util;

import java.io.*;

import jcuda.CudaException;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.*;

public class JCudaHelper {
    public static CUmodule module = new CUmodule();
    static boolean INIT = false;

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
        String outputFileName = cuFileName.substring(0, dotIndex) + "." + targetFileType.toLowerCase();
        File ptxFile = new File(outputFileName);
        if (ptxFile.exists() && !forceRebuild) {
            return outputFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new CudaException("Input file not found: " + cuFileName + " (" + cuFile.getAbsolutePath() + ")");
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        StringBuilder command = new StringBuilder("nvcc ");
        command.append(modelString).append(" ");
        command.append("-").append(targetFileType).append(" ");
        for (String a : nvccArguments) {
            command.append(a).append(" ");
        }
        command.append("\"").append(cuFileName).append("\" -o \"").append(outputFileName).append("\"");

        try {
            System.out.println(command);
            Process process = Runtime.getRuntime().exec(command.toString());

            String errorMessage = new String(toByteArray(process.getErrorStream()));
            int exitValue;
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

        return outputFileName;
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
        String kernelData;
        boolean rebuild = false;

        String runDir = System.getProperty("user.dir") + "/run/";

        try {
            InputStream kernelInput = JCudaHelper.class.getClassLoader().getResourceAsStream(cuFileName);
            if (kernelInput == null) {
                throw new NullPointerException("Cannot find file " + cuFileName);
            }
            kernelData = new String(kernelInput.readAllBytes());
            File globalKernelFile = new File(runDir + cuFileName);
            if (globalKernelFile.exists()) {
                InputStream reader = new FileInputStream(globalKernelFile);
                if (!kernelData.equals(new String(reader.readAllBytes()))) {
                    OutputStream writer = new FileOutputStream(globalKernelFile);
                    writer.write(kernelData.getBytes());
                    rebuild = true;
                }
            } else {
                boolean madeDirs = new File(runDir + cuFileName.substring(0, cuFileName.lastIndexOf("/"))).mkdirs();
                boolean madeFile = globalKernelFile.createNewFile();
                OutputStream writer = new FileOutputStream(globalKernelFile);
                writer.write(kernelData.getBytes());
                rebuild = true;
            }
        } catch (Exception e) {
            e.printStackTrace();
//			throw new Exception("Unable to save " + cuFileName +  " to C:\\com.oroarmor.neural_network.oroarmor\\" + cuFileName);
        }

        String result;
        try {
            result = invokeNvcc(runDir + cuFileName, "cubin", rebuild,
                    "-dlink", "-arch=sm_" + computeComputeCapability());
        } catch (Exception e) {
            result = prepareCubinFile(cuFileName);
        }
        return result;
    }

    public static String prepareCubinFile(String cuFileName) {
        return invokeNvcc(cuFileName, "cubin", true,
                "-dlink", "-arch=sm_" + computeComputeCapability());
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
        int[] majorArray = {0};
        int[] minorArray = {0};
        cuDeviceGetAttribute(majorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(minorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        int major = majorArray[0];
        int minor = minorArray[0];
        return major * 10 + minor;
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