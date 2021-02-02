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

package com.oroarmor.neural_network.util;

import java.io.*;

import jcuda.CudaException;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.*;

/**
 * A helper with JCuda features
 * @author OroArmor
 */
public class JCudaHelper {
    /**
     * The cuda module
     */
    public static CUmodule module = new CUmodule();

    /**
     * True if JCuda has been initialized
     */
    protected static boolean INIT = false;

    /**
     * Invokes the nvcc compiler on the cuFileName
     * @param cuFileName The file
     * @param targetFileType The target type
     * @param forceRebuild Force the rebuild of the file
     * @param nvccArguments Extra nvcc arguments
     * @return The output file name
     */
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
            Process process = Runtime.getRuntime().exec(command.toString());

            String errorMessage = new String(process.getErrorStream().readAllBytes());
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

    /**
     * Prepares a cubin file for the input
     * @param cuFileName The cu file
     * @return The output file name
     */
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

    private static String prepareCubinFile(String cuFileName) {
        return invokeNvcc(cuFileName, "cubin", true,
                "-dlink", "-arch=sm_" + computeComputeCapability());
    }

    /**
     * Creates a new ptx file only if the output does not exist already
     * @param cuFileName The input file
     * @return the output file name
     */
    public static String prepareDefaultPtxFile(String cuFileName) {
        return invokeNvcc(cuFileName, "ptx", false);
    }

    /**
     * Creates a new ptx file only
     * @param cuFileName The input file
     * @return the output file name
     */
    public static String preparePtxFile(String cuFileName) {
        return invokeNvcc(cuFileName, "ptx", true);
    }

    /**
     * Computes the capability of the current device
     * @return The compute capability
     */
    private static int computeComputeCapability() {
        CUdevice device = new CUdevice();
        int status = cuCtxGetDevice(device);
        if (status != CUresult.CUDA_SUCCESS) {
            throw new CudaException(CUresult.stringFor(status));
        }
        return computeComputeCapability(device);
    }

    /**
     * Computes the capability of the current device
     * @param device The cuda device
     * @return The compute capability
     */
    private static int computeComputeCapability(CUdevice device) {
        int[] majorArray = {0};
        int[] minorArray = {0};
        cuDeviceGetAttribute(majorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(minorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        int major = majorArray[0];
        int minor = minorArray[0];
        return major * 10 + minor;
    }

    /**
     * Initialize JCuda. This must be run before any other JCuda features are run
     * @param setExceptions True if exceptions are to be thrown
     */
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