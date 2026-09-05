/** Console entry point for the minimal Java/JVM OCR example. */
public final class OcrDemo {
    private OcrDemo() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 4) {
            System.err.println("Usage: OcrDemo <models-directory> <image-file> [workers] [cls]");
            System.exit(2);
        }

        String modelDirectory = args[0];
        String imagePath = args[1];
        int workers = args.length >= 3 ? Integer.parseInt(args[2]) : 0;
        boolean useCls = args.length >= 4 && Boolean.parseBoolean(args[3]);

        long started = System.nanoTime();
        String[] lines;
        try (NativeOcr ocr = new NativeOcr(modelDirectory, useCls, workers)) {
            lines = ocr.recognizeFile(imagePath);
            // Explicitly exercise idempotent close; try-with-resources closes
            // the same engine once more at scope exit.
            ocr.close();
            ocr.close();
        }
        long elapsedMs = (System.nanoTime() - started) / 1_000_000L;

        System.out.println("image: " + imagePath);
        System.out.println("lines: " + lines.length);
        System.out.println("Java end-to-end elapsed: " + elapsedMs + " ms");
        for (int index = 0; index < lines.length; ++index) {
            System.out.printf("[%02d] %s%n", index + 1, lines[index]);
        }
    }
}
