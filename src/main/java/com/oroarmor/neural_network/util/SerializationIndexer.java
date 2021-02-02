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

/**
 * Holds the serialization indexes
 * @author OroArmor
 */
public final class SerializationIndexer {
    public static final long NETWORK_ID_HEADER = 0L;
    public static final long NEURAL_NETWORK_ID = NETWORK_ID_HEADER + 1L;
    public static final long AUTO_ENCODER_ID = NETWORK_ID_HEADER + 2L;

    public static final long LAYER_ID_HEADER = 10L;
    public static final long FEED_FORWARD_LAYER_ID = LAYER_ID_HEADER + 1L;
    public static final long KEEP_POSITIVE_LAYER_ID = LAYER_ID_HEADER + 2L;
    public static final long SOFT_MAX_LAYER_ID = LAYER_ID_HEADER + 3L;

    public static final long MATRIX_ID_HEADER = 20L;
    public static final long CPU_MATRIX_ID = MATRIX_ID_HEADER + 1L;
    public static final long GPU_MATRIX_ID = MATRIX_ID_HEADER + 2L;
}
