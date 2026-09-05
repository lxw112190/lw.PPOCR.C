# The SDK consumer rules keep NativeBridge and NativeOcrPacket for JNI.
# Keep the Java Activity entry point and image loader for readable Preview
# stack traces; no additional native rules are needed in this application.
-keep class com.lxw112190.ppocr.javademo.MainActivity { *; }
-keep class com.lxw112190.ppocr.javademo.JavaImageLoader { *; }
