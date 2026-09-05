package com.lxw112190.ppocr.javademo;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;

import androidx.exifinterface.media.ExifInterface;

import java.io.IOException;
import java.io.InputStream;

/**
 * Loads a content:// image safely for the native OCR API.
 *
 * <p>The loader deliberately performs a bounds-only pass first. A null
 * Bitmap from that pass is expected because {@code inJustDecodeBounds} is
 * enabled; the bounds values, not the return value, determine success.</p>
 */
public final class JavaImageLoader {
    private static final long MAX_DECODE_PIXELS = 12_000_000L;

    private JavaImageLoader() {
    }

    public static Bitmap load(Context context, Uri uri) throws IOException {
        ContentResolver resolver = context.getContentResolver();
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;

        try (InputStream input = open(resolver, uri, "无法打开所选图片")) {
            BitmapFactory.decodeStream(input, null, bounds);
        }

        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("图片格式不受支持或图片已损坏");
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight);
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap decoded;
        try (InputStream input = open(resolver, uri, "无法重新打开所选图片")) {
            decoded = BitmapFactory.decodeStream(input, null, options);
        }
        if (decoded == null) {
            throw new IOException("图片解码失败，格式可能不受当前设备支持");
        }

        Bitmap normalized = decoded;
        try {
            int orientation = readOrientation(resolver, uri);
            normalized = applyOrientation(normalized, orientation);
            if (normalized != decoded && !decoded.isRecycled()) {
                decoded.recycle();
            }

            normalized = ensureArgb8888(normalized);
            normalized = scaleToPixelLimit(normalized);
            return normalized;
        } catch (IOException | RuntimeException error) {
            recycleDistinct(normalized, decoded);
            throw error;
        }
    }

    private static InputStream open(
            ContentResolver resolver,
            Uri uri,
            String message
    ) throws IOException {
        InputStream input = resolver.openInputStream(uri);
        if (input == null) {
            throw new IOException(message);
        }
        return input;
    }

    private static int calculateSampleSize(int width, int height) {
        int sample = 1;
        while ((width / (long) sample) * (height / (long) sample) > MAX_DECODE_PIXELS) {
            sample *= 2;
        }
        return sample;
    }

    private static int readOrientation(ContentResolver resolver, Uri uri) throws IOException {
        try (InputStream input = open(resolver, uri, "无法读取图片方向")) {
            return new ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );
        }
    }

    private static Bitmap applyOrientation(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180f);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setScale(1f, -1f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90f);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90f);
                break;
            default:
                return bitmap;
        }

        return Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
    }

    private static Bitmap ensureArgb8888(Bitmap bitmap) throws IOException {
        if (bitmap.getConfig() == Bitmap.Config.ARGB_8888) {
            return bitmap;
        }
        Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        if (converted == null) {
            throw new IOException("无法转换图片像素格式");
        }
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return converted;
    }

    private static Bitmap scaleToPixelLimit(Bitmap bitmap) {
        long pixels = bitmap.getWidth() * (long) bitmap.getHeight();
        if (pixels <= MAX_DECODE_PIXELS) {
            return bitmap;
        }

        double scale = Math.sqrt(MAX_DECODE_PIXELS / (double) pixels);
        int width = Math.max(1, (int) Math.ceil(bitmap.getWidth() * scale));
        int height = Math.max(1, (int) Math.ceil(bitmap.getHeight() * scale));
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, width, height, true);
        if (scaled != bitmap && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return scaled;
    }

    private static void recycleDistinct(Bitmap first, Bitmap second) {
        if (first != null && !first.isRecycled()) {
            first.recycle();
        }
        if (second != null && second != first && !second.isRecycled()) {
            second.recycle();
        }
    }
}
