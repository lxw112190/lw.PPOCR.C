import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

/**
 * Minimal desktop Java/JVM wrapper over the lw.PPOCR.C C ABI.
 *
 * <p>This example deliberately keeps the public surface small: one engine is
 * used serially, recognition returns only the ordered text lines, and the JNI
 * library is loaded from {@code java.library.path}. It is not an Android or
 * Maven SDK.</p>
 */
public final class NativeOcr implements AutoCloseable {
    public static final int READING_ORDER_HORIZONTAL_LTR = 0;
    public static final int READING_ORDER_VERTICAL_RTL = 1;
    public static final int READING_ORDER_VERTICAL_LTR = 2;

    private static final long MAX_IMAGE_PIXELS = 40_000_000L;

    static {
        System.loadLibrary("lw_ppocr_java");
    }

    private long handle;

    public NativeOcr(String modelDirectory) {
        this(modelDirectory, false, 0);
    }

    public NativeOcr(String modelDirectory, boolean useCls, int workerCount) {
        this(
                modelPath(modelDirectory, "det.lwm"),
                modelPath(modelDirectory, "cls.lwm"),
                modelPath(modelDirectory, "rec.lwm"),
                modelPath(modelDirectory, "ppocr_keys.txt"),
                useCls,
                workerCount);
    }

    public NativeOcr(
            String detector,
            String classifier,
            String recognizer,
            String dictionary,
            boolean useCls,
            int workerCount) {
        if (detector == null || recognizer == null || dictionary == null) {
            throw new NullPointerException("model paths must not be null");
        }
        handle = nativeCreate(
                detector,
                classifier,
                recognizer,
                dictionary,
                useCls,
                workerCount);
        if (handle == 0L) {
            throw new IllegalStateException("Unable to create OCR engine");
        }
    }

    public synchronized String[] recognizeFile(String imagePath) throws IOException {
        if (imagePath == null) {
            throw new NullPointerException("imagePath");
        }
        BufferedImage source = ImageIO.read(new File(imagePath));
        if (source == null) {
            throw new IOException("Unsupported or invalid image: " + imagePath);
        }
        return recognize(source);
    }

    public synchronized String[] recognize(BufferedImage source) {
        ensureOpen();
        if (source == null) {
            throw new NullPointerException("image");
        }
        BufferedImage image = toBgr(source);
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        return nativeRecognize(handle, pixels, width, height, width * 3);
    }

    public synchronized void setReadingOrder(int readingOrder) {
        ensureOpen();
        if (readingOrder < READING_ORDER_HORIZONTAL_LTR ||
                readingOrder > READING_ORDER_VERTICAL_LTR) {
            throw new IllegalArgumentException("unsupported reading order: " + readingOrder);
        }
        nativeSetReadingOrder(handle, readingOrder);
    }

    @Override
    public synchronized void close() {
        if (handle != 0L) {
            nativeDestroy(handle);
            handle = 0L;
        }
    }

    private void ensureOpen() {
        if (handle == 0L) {
            throw new IllegalStateException("OCR engine is closed");
        }
    }

    private static String modelPath(String directory, String name) {
        if (directory == null) {
            throw new NullPointerException("modelDirectory");
        }
        Path path = Paths.get(directory).resolve(name);
        return path.toString();
    }

    private static BufferedImage toBgr(BufferedImage source) {
        int width = source.getWidth();
        int height = source.getHeight();
        long pixels = (long) width * height;
        if (width <= 0 || height <= 0 || pixels > MAX_IMAGE_PIXELS) {
            throw new IllegalArgumentException(
                    "Decoded image is too large: " + width + "x" + height);
        }

        /* Always create the target so the byte array has a compact,
         * width*3 scanline. This avoids relying on a caller's raster offset or
         * stride even when the source already uses TYPE_3BYTE_BGR. */
        BufferedImage target = new BufferedImage(
                width,
                height,
                BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D graphics = target.createGraphics();
        try {
            graphics.drawImage(source, 0, 0, null);
        } finally {
            graphics.dispose();
        }
        return target;
    }

    private static native long nativeCreate(
            String detector,
            String classifier,
            String recognizer,
            String dictionary,
            boolean useCls,
            int workerCount);

    private static native String[] nativeRecognize(
            long handle,
            byte[] bgr,
            int width,
            int height,
            int stride);

    private static native void nativeSetReadingOrder(long handle, int readingOrder);

    private static native void nativeDestroy(long handle);
}
