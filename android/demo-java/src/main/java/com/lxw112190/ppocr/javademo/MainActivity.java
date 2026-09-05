package com.lxw112190.ppocr.javademo;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;

import com.lxw112190.ppocr.LwPpocrEngine;
import com.lxw112190.ppocr.OcrLine;
import com.lxw112190.ppocr.OcrOptions;
import com.lxw112190.ppocr.OcrResult;
import com.lxw112190.ppocr.ReadingOrder;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal Java integration sample for the lw.PPOCR Android SDK.
 *
 * <p>All SDK calls, image decoding, and Bitmap ownership transitions happen
 * on one worker. UI state and views remain on the main thread.</p>
 */
public final class MainActivity extends ComponentActivity {
    private enum UiState {
        ENGINE_LOADING,
        READY,
        IMAGE_LOADING,
        IMAGE_READY,
        RECOGNIZING,
        RESULT_READY,
        ERROR
    }

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ActivityResultLauncher<PickVisualMediaRequest> pickImage =
            registerForActivityResult(
                    new ActivityResultContracts.PickVisualMedia(),
                    uri -> {
                        if (uri != null) {
                            prepareImage(uri);
                        }
                    }
            );

    private Button chooseButton;
    private Button runButton;
    private Switch useCls;
    private Spinner readingOrder;
    private ImageView preview;
    private TextView status;
    private TextView statusDetail;
    private TextView resultMeta;
    private TextView resultText;

    private Bitmap currentBitmap;
    private OcrResult currentResult;
    private LwPpocrEngine engine;
    private ReadingOrder desiredReadingOrder = ReadingOrder.HORIZONTAL_LTR;
    private boolean desiredUseCls;
    private boolean engineUseCls;
    private volatile boolean engineReady;
    private volatile boolean destroyed;
    private volatile long operationGeneration;
    private UiState state = UiState.ENGINE_LOADING;
    private String errorMessage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(com.lxw112190.ppocr.javademo.R.layout.activity_main);

        chooseButton = findViewById(R.id.choose_button);
        runButton = findViewById(R.id.run_button);
        useCls = findViewById(R.id.use_cls);
        readingOrder = findViewById(R.id.reading_order);
        preview = findViewById(R.id.preview);
        status = findViewById(R.id.status);
        statusDetail = findViewById(R.id.status_detail);
        resultMeta = findViewById(R.id.result_meta);
        resultText = findViewById(R.id.result_text);

        ArrayAdapter<String> orderAdapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                new String[]{"标准横排", "古籍竖排（右→左）", "竖排（左→右）"}
        );
        orderAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        readingOrder.setAdapter(orderAdapter);

        chooseButton.setOnClickListener(view ->
                pickImage.launch(new PickVisualMediaRequest(
                        ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE
                )));
        runButton.setOnClickListener(view -> recognizeCurrentImage());

        useCls.setOnCheckedChangeListener((button, checked) -> {
            if (desiredUseCls == checked) {
                return;
            }
            desiredUseCls = checked;
            clearResult();
            setImageReadyState();
        });

        readingOrder.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(
                    android.widget.AdapterView<?> parent,
                    View view,
                    int position,
                    long id
            ) {
                ReadingOrder selected = position == 1
                        ? ReadingOrder.VERTICAL_RTL
                        : position == 2
                        ? ReadingOrder.VERTICAL_LTR
                        : ReadingOrder.HORIZONTAL_LTR;
                if (desiredReadingOrder != selected) {
                    desiredReadingOrder = selected;
                    clearResult();
                    setImageReadyState();
                }
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {
                // The first entry is always the default.
            }
        });

        renderState();
        initializeEngine();
    }

    private void initializeEngine() {
        state = UiState.ENGINE_LOADING;
        errorMessage = null;
        renderState();
        final boolean useClsAtStart = desiredUseCls;
        final ReadingOrder orderAtStart = desiredReadingOrder;

        worker.execute(() -> {
            try {
                LwPpocrEngine created = LwPpocrEngine.createBlocking(
                        getApplicationContext(),
                        new OcrOptions(useClsAtStart, 2, 20_000_000L, orderAtStart)
                );
                if (destroyed) {
                    created.close();
                    return;
                }
                engine = created;
                engineUseCls = useClsAtStart;
                engineReady = true;
                postToMain(() -> {
                    if (destroyed) {
                        return;
                    }
                    state = currentBitmap == null ? UiState.READY : UiState.IMAGE_READY;
                    renderState();
                });
            } catch (Exception error) {
                postError(error);
            }
        });
    }

    private void prepareImage(Uri uri) {
        state = UiState.IMAGE_LOADING;
        errorMessage = null;
        clearResult();
        renderState();
        final long token = ++operationGeneration;

        worker.execute(() -> {
            try {
                Bitmap loaded = JavaImageLoader.load(this, uri);
                if (destroyed || token != operationGeneration) {
                    recycle(loaded);
                    return;
                }
                postToMain(() -> {
                    if (destroyed || token != operationGeneration) {
                        recycle(loaded);
                        return;
                    }
                    replaceBitmap(loaded);
                    state = engineReady ? UiState.IMAGE_READY : UiState.ERROR;
                    if (!engineReady) {
                        errorMessage = "引擎仍未就绪，请稍后重试";
                    }
                    renderState();
                });
            } catch (Exception error) {
                postError(error);
            }
        });
    }

    private void recognizeCurrentImage() {
        final Bitmap bitmap = currentBitmap;
        if (bitmap == null || !engineReady) {
            return;
        }

        state = UiState.RECOGNIZING;
        errorMessage = null;
        renderState();
        final long token = ++operationGeneration;
        final boolean useClsAtStart = desiredUseCls;
        final ReadingOrder orderAtStart = desiredReadingOrder;

        worker.execute(() -> {
            try {
                LwPpocrEngine active = ensureEngineForCls(useClsAtStart, orderAtStart);
                active.setReadingOrder(orderAtStart);
                OcrResult result = active.recognizeBlocking(bitmap);
                postToMain(() -> {
                    if (destroyed || token != operationGeneration) {
                        return;
                    }
                    currentResult = result;
                    state = UiState.RESULT_READY;
                    renderResult(result);
                    renderState();
                });
            } catch (Exception error) {
                postError(error);
            }
        });
    }

    private LwPpocrEngine ensureEngineForCls(
            boolean useCls,
            ReadingOrder order
    ) throws Exception {
        if (engine != null && engineUseCls == useCls) {
            return engine;
        }

        if (engine != null) {
            engine.close();
            engine = null;
        }
        engineReady = false;

        LwPpocrEngine fresh = LwPpocrEngine.createBlocking(
                getApplicationContext(),
                new OcrOptions(useCls, 2, 20_000_000L, order)
        );
        engine = fresh;
        engineUseCls = useCls;
        engineReady = true;
        return fresh;
    }

    private void replaceBitmap(Bitmap bitmap) {
        Bitmap old = currentBitmap;
        currentBitmap = null;
        preview.setImageDrawable(null);
        recycle(old);
        currentBitmap = bitmap;
        preview.setImageBitmap(bitmap);
    }

    private void clearResult() {
        currentResult = null;
        resultText.setText("");
        resultMeta.setText(currentBitmap == null ? "等待图片" : "等待识别");
    }

    private void renderResult(OcrResult result) {
        StringBuilder text = new StringBuilder();
        for (OcrLine line : result.getLines()) {
            if (text.length() > 0) {
                text.append('\n');
            }
            text.append(line.getIndex() + 1).append(". ").append(line.getText());
        }
        resultMeta.setText(
                result.getLines().size() + " 行 · OCR耗时 " + result.getElapsedMs() + " ms"
        );
        resultText.setText(text.toString());
    }

    private void setImageReadyState() {
        if (state == UiState.ENGINE_LOADING || state == UiState.IMAGE_LOADING ||
                state == UiState.RECOGNIZING) {
            return;
        }
        state = currentBitmap == null
                ? (engineReady ? UiState.READY : UiState.ERROR)
                : (engineReady ? UiState.IMAGE_READY : UiState.ERROR);
        renderState();
    }

    private void renderState() {
        boolean busy = state == UiState.ENGINE_LOADING ||
                state == UiState.IMAGE_LOADING ||
                state == UiState.RECOGNIZING;
        chooseButton.setEnabled(!busy);
        runButton.setEnabled(!busy && engineReady && currentBitmap != null);
        useCls.setEnabled(!busy && engineReady);
        readingOrder.setEnabled(!busy && engineReady);

        switch (state) {
            case ENGINE_LOADING:
                setStatus("正在准备引擎…", "首次启动会准备离线模型");
                break;
            case READY:
                setStatus("引擎已就绪", "ARM64 Native · 可以选择图片");
                break;
            case IMAGE_LOADING:
                setStatus("正在准备图片…", "正在读取并规范化图片");
                break;
            case IMAGE_READY:
                setStatus("图片已准备好", "点击开始识别");
                break;
            case RECOGNIZING:
                setStatus("正在识别…", "图片不会离开本机");
                break;
            case RESULT_READY:
                setStatus("识别完成", "可以继续选择图片或调整设置");
                break;
            case ERROR:
                setStatus("处理失败", errorMessage == null ? "请重试" : errorMessage);
                break;
        }
    }

    private void setStatus(String title, String detail) {
        status.setText(title);
        statusDetail.setText(detail);
    }

    private void postError(Exception error) {
        postToMain(() -> {
            if (destroyed) {
                return;
            }
            engineReady = engine != null;
            state = UiState.ERROR;
            errorMessage = error.getMessage() == null
                    ? error.getClass().getSimpleName()
                    : error.getMessage();
            renderState();
        });
    }

    private void postToMain(Runnable action) {
        if (!destroyed) {
            mainHandler.post(action);
        }
    }

    private static void recycle(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        operationGeneration++;
        preview.setImageDrawable(null);
        final Bitmap bitmapAtDestroy = currentBitmap;
        currentBitmap = null;

        worker.execute(() -> {
            if (engine != null) {
                engine.close();
                engine = null;
            }
            engineReady = false;
            recycle(bitmapAtDestroy);
            worker.shutdown();
        });
        super.onDestroy();
    }
}
