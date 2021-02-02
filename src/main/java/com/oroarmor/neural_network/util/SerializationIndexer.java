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
