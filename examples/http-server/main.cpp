#ifndef LW_HTTP_THREAD_POOL_COUNT
#  if (defined(_WIN32) && !defined(_WIN64)) || \
      (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4)
#    define LW_HTTP_THREAD_POOL_COUNT 2
#  else
#    define LW_HTTP_THREAD_POOL_COUNT 8
#  endif
#endif
#define CPPHTTPLIB_THREAD_POOL_COUNT LW_HTTP_THREAD_POOL_COUNT

#include "httplib.h"
#include "lw_infer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <unistd.h>
#else
#  include <unistd.h>
#endif

namespace {

const uint64_t kMaxImagePixels = UINT64_C(40000000);
const size_t kMaxWebPageBytes = 4u * 1024u * 1024u;

bool AllocationFits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= static_cast<uint64_t>(SIZE_MAX / element_size);
}

uint32_t DefaultOcrWorkers() {
    lw_ocr_options options;
    lw_ocr_options_init(&options);
    return options.worker_count;
}

struct Config {
    std::string host;
    int port;
    std::string models;
    std::string web_root;
    bool use_classifier;
    uint32_t ocr_workers;
    uint32_t rec_max_width;

    Config() : host("127.0.0.1"), port(8787), use_classifier(true),
               ocr_workers(DefaultOcrWorkers()), rec_max_width(320u) {}
};

struct BgrImage {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> pixels;

    BgrImage() : width(0u), height(0u) {}
};

struct OcrOutput {
    uint32_t detected_count;
    std::vector<lw_ocr_line> lines;
    std::string text;

    OcrOutput() : detected_count(0u) {}
};

class ApiError : public std::runtime_error {
public:
    ApiError(int status, const std::string& code, const std::string& message)
        : std::runtime_error(message), status_(status), code_(code) {}
    int status() const { return status_; }
    const std::string& code() const { return code_; }

private:
    int status_;
    std::string code_;
};

std::mutex g_log_mutex;
std::atomic<uint64_t> g_request_sequence(0u);

void Log(const std::string& message) {
    std::lock_guard<std::mutex> guard(g_log_mutex);
    std::cout << message << std::endl;
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (size_t index = 0u; index < value.size(); ++index) {
        const unsigned char character =
            static_cast<unsigned char>(value[index]);
        switch (character) {
        case '\"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20u) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned int>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
            break;
        }
    }
    return output.str();
}

std::string MakeRequestId() {
    const uint64_t ticks = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const uint64_t sequence = ++g_request_sequence;
    std::ostringstream output;
    output << std::hex << ticks << '-' << sequence;
    return output.str();
}

std::string ParentPath(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0u) return path.substr(0u, 1u);
    return path.substr(0u, slash);
}

std::string JoinPath(const std::string& parent, const std::string& child) {
    if (parent.empty()) return child;
    if (parent == ".") return "./" + child;
    const char last = parent[parent.size() - 1u];
    return parent + ((last == '/' || last == '\\') ? "" : "/") + child;
}

#if defined(_WIN32)
std::string WideToUtf8(const wchar_t* value) {
    if (value == NULL || *value == L'\0') return std::string();
    const int count = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, NULL, 0, NULL, NULL);
    if (count <= 0) throw std::runtime_error("UTF-16 to UTF-8 conversion failed");
    std::vector<char> bytes(static_cast<size_t>(count));
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
            &bytes[0], count, NULL, NULL) <= 0) {
        throw std::runtime_error("UTF-16 to UTF-8 conversion failed");
    }
    return std::string(&bytes[0]);
}

std::wstring Utf8ToWide(const std::string& value) {
    if (value.empty()) return std::wstring();
    const int count = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, NULL, 0);
    if (count <= 0) throw std::runtime_error("UTF-8 path conversion failed");
    std::vector<wchar_t> characters(static_cast<size_t>(count));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1,
            &characters[0], count) <= 0) {
        throw std::runtime_error("UTF-8 path conversion failed");
    }
    return std::wstring(&characters[0]);
}
#endif

std::string ExecutableDirectory(const char* argv0) {
#if defined(_WIN32)
    std::vector<wchar_t> path(32768u);
    const DWORD count = GetModuleFileNameW(
        NULL, &path[0], static_cast<DWORD>(path.size()));
    if (count != 0u && count < path.size()) {
        path[count] = L'\0';
        return ParentPath(WideToUtf8(&path[0]));
    }
#elif defined(__APPLE__)
    uint32_t size = 0u;
    (void)_NSGetExecutablePath(NULL, &size);
    if (size != 0u) {
        std::vector<char> path(size + 1u, '\0');
        if (_NSGetExecutablePath(&path[0], &size) == 0) {
            return ParentPath(&path[0]);
        }
    }
#else
    std::vector<char> path(4096u, '\0');
    const ssize_t count = readlink("/proc/self/exe", &path[0], path.size() - 1u);
    if (count > 0) {
        path[static_cast<size_t>(count)] = '\0';
        return ParentPath(&path[0]);
    }
#endif
    return ParentPath(argv0 == NULL ? "." : argv0);
}

std::string ReadFileUtf8(const std::string& path, size_t maximum_size) {
    FILE* file = NULL;
#if defined(_WIN32)
    const std::wstring wide = Utf8ToWide(path);
    if (_wfopen_s(&file, wide.c_str(), L"rb") != 0) file = NULL;
#else
    file = std::fopen(path.c_str(), "rb");
#endif
    if (file == NULL) throw std::runtime_error("unable to open file: " + path);
    std::vector<char> data;
    char buffer[8192];
    for (;;) {
        const size_t count = std::fread(buffer, 1u, sizeof(buffer), file);
        if (count != 0u) {
            if (data.size() > maximum_size - count) {
                std::fclose(file);
                throw std::runtime_error("file is too large: " + path);
            }
            data.insert(data.end(), buffer, buffer + count);
        }
        if (count < sizeof(buffer)) {
            if (std::ferror(file) != 0) {
                std::fclose(file);
                throw std::runtime_error("unable to read file: " + path);
            }
            break;
        }
    }
    std::fclose(file);
    return std::string(data.begin(), data.end());
}

bool IsSpace(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
        value == '\f' || value == '\v';
}

std::string ReadPpmToken(const std::string& bytes, size_t* offset) {
    if (offset == NULL) throw ApiError(400, "bad_request", "invalid PPM parser state");
    for (;;) {
        while (*offset < bytes.size() &&
               IsSpace(static_cast<unsigned char>(bytes[*offset]))) ++(*offset);
        if (*offset >= bytes.size() || bytes[*offset] != '#') break;
        while (*offset < bytes.size() && bytes[*offset] != '\n') ++(*offset);
    }
    const size_t begin = *offset;
    while (*offset < bytes.size() &&
           !IsSpace(static_cast<unsigned char>(bytes[*offset]))) ++(*offset);
    if (*offset == begin || *offset >= bytes.size()) {
        throw ApiError(400, "bad_request", "invalid P6 PPM header");
    }
    const std::string token = bytes.substr(begin, *offset - begin);
    const unsigned char delimiter = static_cast<unsigned char>(bytes[*offset]);
    ++(*offset);
    if (delimiter == '\r' && *offset < bytes.size() && bytes[*offset] == '\n') {
        ++(*offset);
    }
    return token;
}

uint32_t ParsePositiveU32(const std::string& value, const char* field) {
    if (value.empty()) throw ApiError(400, "bad_request", std::string("invalid ") + field);
    uint64_t parsed = 0u;
    for (size_t index = 0u; index < value.size(); ++index) {
        const unsigned char character = static_cast<unsigned char>(value[index]);
        if (character < '0' || character > '9') {
            throw ApiError(400, "bad_request", std::string("invalid ") + field);
        }
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (parsed > (UINT32_MAX - digit) / 10u) {
            throw ApiError(400, "bad_request", std::string("invalid ") + field);
        }
        parsed = parsed * 10u + digit;
    }
    if (parsed == 0u) throw ApiError(400, "bad_request", std::string("invalid ") + field);
    return static_cast<uint32_t>(parsed);
}

BgrImage DecodeP6Ppm(const std::string& bytes) {
    size_t offset = 0u;
    if (ReadPpmToken(bytes, &offset) != "P6") {
        throw ApiError(415, "unsupported_media_type",
            "server accepts binary P6 PPM; the browser Demo converts JPEG/PNG before upload");
    }
    BgrImage image;
    image.width = ParsePositiveU32(ReadPpmToken(bytes, &offset), "PPM width");
    image.height = ParsePositiveU32(ReadPpmToken(bytes, &offset), "PPM height");
    if (ParsePositiveU32(ReadPpmToken(bytes, &offset), "PPM max value") != 255u) {
        throw ApiError(400, "bad_request", "P6 PPM max value must be 255");
    }
    const uint64_t pixels = static_cast<uint64_t>(image.width) * image.height;
    if (pixels > kMaxImagePixels || pixels > SIZE_MAX / 3u) {
        throw ApiError(413, "image_too_large", "decoded image exceeds 40,000,000 pixels");
    }
    const size_t byte_count = static_cast<size_t>(pixels * 3u);
    if (offset > bytes.size() || bytes.size() - offset != byte_count) {
        throw ApiError(400, "bad_request", "P6 PPM pixel payload length is invalid");
    }
    image.pixels.resize(byte_count);
    for (size_t index = 0u; index < byte_count; index += 3u) {
        image.pixels[index] = static_cast<uint8_t>(bytes[offset + index + 2u]);
        image.pixels[index + 1u] = static_cast<uint8_t>(bytes[offset + index + 1u]);
        image.pixels[index + 2u] = static_cast<uint8_t>(bytes[offset + index]);
    }
    return image;
}

uint16_t ReadLe16(const std::string& bytes, size_t offset, const char* field) {
    if (offset > bytes.size() || bytes.size() - offset < 2u) {
        throw ApiError(400, "bad_request", std::string("invalid BMP ") + field);
    }
    return static_cast<uint16_t>(
        static_cast<uint8_t>(bytes[offset]) |
        (static_cast<uint16_t>(static_cast<uint8_t>(bytes[offset + 1u])) << 8u));
}

uint32_t ReadLe32(const std::string& bytes, size_t offset, const char* field) {
    if (offset > bytes.size() || bytes.size() - offset < 4u) {
        throw ApiError(400, "bad_request", std::string("invalid BMP ") + field);
    }
    return static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 1u])) << 8u) |
        (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 2u])) << 16u) |
        (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 3u])) << 24u);
}

BgrImage DecodeBmp24(const std::string& bytes) {
    if (bytes.size() < 54u || bytes[0] != 'B' || bytes[1] != 'M') {
        throw ApiError(400, "bad_request", "invalid BMP header");
    }
    const uint32_t pixel_offset = ReadLe32(bytes, 10u, "pixel offset");
    const uint32_t dib_size = ReadLe32(bytes, 14u, "DIB header");
    const uint32_t raw_width = ReadLe32(bytes, 18u, "width");
    const uint32_t raw_height = ReadLe32(bytes, 22u, "height");
    const int64_t signed_width = raw_width <= INT32_MAX
        ? static_cast<int64_t>(raw_width)
        : static_cast<int64_t>(raw_width) - INT64_C(4294967296);
    const int64_t signed_height = raw_height <= INT32_MAX
        ? static_cast<int64_t>(raw_height)
        : static_cast<int64_t>(raw_height) - INT64_C(4294967296);
    const uint16_t planes = ReadLe16(bytes, 26u, "planes");
    const uint16_t bits_per_pixel = ReadLe16(bytes, 28u, "bit depth");
    const uint32_t compression = ReadLe32(bytes, 30u, "compression");
    if (dib_size < 40u || dib_size > bytes.size() - 14u ||
        pixel_offset < 14u + static_cast<uint64_t>(dib_size) ||
        planes != 1u || bits_per_pixel != 24u ||
        compression != 0u || signed_width <= 0 || signed_height == 0) {
        throw ApiError(415, "unsupported_media_type",
            "server accepts uncompressed 24-bit BMP images");
    }
    BgrImage image;
    image.width = static_cast<uint32_t>(signed_width);
    image.height = static_cast<uint32_t>(
        signed_height < 0 ? -signed_height : signed_height);
    const uint64_t pixels = static_cast<uint64_t>(image.width) * image.height;
    if (pixels > kMaxImagePixels || !AllocationFits(pixels, 3u)) {
        throw ApiError(413, "image_too_large", "decoded image exceeds 40,000,000 pixels");
    }
    const uint64_t row_bytes =
        (static_cast<uint64_t>(image.width) * 3u + 3u) & ~UINT64_C(3);
    const uint64_t payload_bytes = row_bytes * image.height;
    if (pixel_offset > bytes.size() || payload_bytes > bytes.size() - pixel_offset) {
        throw ApiError(400, "bad_request", "BMP pixel payload length is invalid");
    }
    image.pixels.resize(static_cast<size_t>(pixels * 3u));
    for (uint32_t y = 0u; y < image.height; ++y) {
        const uint32_t source_y =
            signed_height < 0 ? y : image.height - 1u - y;
        const size_t source_offset =
            static_cast<size_t>(pixel_offset + row_bytes * source_y);
        const size_t destination_offset =
            static_cast<size_t>(image.width) * y * 3u;
        std::memcpy(&image.pixels[destination_offset],
            bytes.data() + source_offset, static_cast<size_t>(image.width) * 3u);
    }
    return image;
}

int Base64Value(unsigned char character) {
    if (character >= 'A' && character <= 'Z') return character - 'A';
    if (character >= 'a' && character <= 'z') return character - 'a' + 26;
    if (character >= '0' && character <= '9') return character - '0' + 52;
    if (character == '+') return 62;
    if (character == '/') return 63;
    return -1;
}

std::string DecodeBase64(std::string value) {
    const size_t comma = value.find(',');
    if (value.compare(0u, 5u, "data:") == 0 && comma != std::string::npos) {
        value.erase(0u, comma + 1u);
    }
    value.erase(std::remove_if(value.begin(), value.end(),
        [](char character) {
            return IsSpace(static_cast<unsigned char>(character));
        }), value.end());
    if (value.empty() || value.size() % 4u != 0u) {
        throw ApiError(400, "bad_request", "imageBase64 is invalid");
    }
    std::string decoded;
    decoded.reserve(value.size() / 4u * 3u);
    for (size_t index = 0u; index < value.size(); index += 4u) {
        const bool final_group = index + 4u == value.size();
        const int a = Base64Value(static_cast<unsigned char>(value[index]));
        const int b = Base64Value(static_cast<unsigned char>(value[index + 1u]));
        const int c = value[index + 2u] == '=' ? -2 :
            Base64Value(static_cast<unsigned char>(value[index + 2u]));
        const int d = value[index + 3u] == '=' ? -2 :
            Base64Value(static_cast<unsigned char>(value[index + 3u]));
        if (a < 0 || b < 0 || c == -1 || d == -1 ||
            (!final_group && (c == -2 || d == -2)) ||
            (c == -2 && d != -2)) {
            throw ApiError(400, "bad_request", "imageBase64 is invalid");
        }
        const uint32_t packed = (static_cast<uint32_t>(a) << 18u) |
            (static_cast<uint32_t>(b) << 12u) |
            (static_cast<uint32_t>(c < 0 ? 0 : c) << 6u) |
            static_cast<uint32_t>(d < 0 ? 0 : d);
        decoded.push_back(static_cast<char>((packed >> 16u) & 0xffu));
        if (c >= 0) decoded.push_back(static_cast<char>((packed >> 8u) & 0xffu));
        if (d >= 0) decoded.push_back(static_cast<char>(packed & 0xffu));
    }
    return decoded;
}

std::string ExtractJsonString(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t offset = json.find(marker);
    if (offset == std::string::npos) return std::string();
    offset += marker.size();
    while (offset < json.size() && IsSpace(static_cast<unsigned char>(json[offset]))) ++offset;
    if (offset >= json.size() || json[offset++] != ':') return std::string();
    while (offset < json.size() && IsSpace(static_cast<unsigned char>(json[offset]))) ++offset;
    if (offset >= json.size() || json[offset++] != '\"') return std::string();
    std::string value;
    while (offset < json.size()) {
        const char character = json[offset++];
        if (character == '\"') return value;
        if (character == '\\') {
            if (offset >= json.size()) break;
            const char escaped = json[offset++];
            if (escaped == '/' || escaped == '\\' || escaped == '\"') {
                value.push_back(escaped);
            } else {
                throw ApiError(400, "bad_request", "imageBase64 contains an invalid JSON escape");
            }
        } else {
            value.push_back(character);
        }
    }
    throw ApiError(400, "bad_request", "JSON string is incomplete");
}

std::string LowerMediaType(const std::string& value) {
    const size_t semicolon = value.find(';');
    std::string type = value.substr(0u, semicolon);
    std::transform(type.begin(), type.end(), type.begin(),
        [](char character) {
            const unsigned char byte = static_cast<unsigned char>(character);
            return static_cast<char>(byte >= 'A' && byte <= 'Z' ? byte + 32u : byte);
        });
    while (!type.empty() && IsSpace(static_cast<unsigned char>(type.front()))) type.erase(type.begin());
    while (!type.empty() && IsSpace(static_cast<unsigned char>(type.back()))) type.pop_back();
    return type;
}

BgrImage ReadRequestImage(const httplib::Request& request) {
    const std::string content_type = LowerMediaType(
        request.get_header_value("Content-Type"));
    if (content_type == "application/json") {
        std::string encoded = ExtractJsonString(request.body, "imageBase64");
        if (encoded.empty()) encoded = ExtractJsonString(request.body, "image_base64");
        if (encoded.empty()) {
            throw ApiError(400, "bad_request", "JSON must contain imageBase64");
        }
        return DecodeP6Ppm(DecodeBase64(encoded));
    }
    if (content_type == "image/x-portable-pixmap" ||
        content_type == "image/x-portable-anymap" ||
        content_type == "application/octet-stream") {
        return DecodeP6Ppm(request.body);
    }
    if (content_type == "image/bmp" || content_type == "image/x-ms-bmp") {
        return DecodeBmp24(request.body);
    }
    throw ApiError(415, "unsupported_media_type",
        "use image/x-portable-pixmap, image/bmp, application/octet-stream, "
        "or JSON/Base64 P6 PPM");
}

class OcrEngine {
public:
    OcrEngine(const Config& config) : handle_(NULL) {
        lw_ocr_options options;
        lw_error error;
        lw_ocr_options_init(&options);
        options.use_direction_classification = config.use_classifier ? 1u : 0u;
        options.worker_count = config.ocr_workers;
        options.recognizer.target_width = config.rec_max_width;
        lw_error_init(&error);
        const std::string detector = JoinPath(config.models, "det.lwm");
        const std::string classifier = JoinPath(config.models, "cls.lwm");
        const std::string recognizer = JoinPath(config.models, "rec.lwm");
        const std::string dictionary = JoinPath(config.models, "ppocr_keys.txt");
        const lw_status status = lw_ocr_create(
            detector.c_str(), config.use_classifier ? classifier.c_str() : NULL,
            recognizer.c_str(), dictionary.c_str(), &options, &handle_, &error);
        if (status != LW_STATUS_OK || handle_ == NULL) {
            throw std::runtime_error(std::string("OCR initialization failed: ") +
                lw_status_string(status) + ": " + error.message);
        }
        lw_ocr_info_init(&info_);
        if (lw_ocr_get_info(handle_, &info_) != LW_STATUS_OK ||
            info_.max_line_capacity == 0u || info_.max_text_capacity == 0u ||
            !AllocationFits(info_.max_line_capacity, sizeof(lw_ocr_line)) ||
            !AllocationFits(info_.max_text_capacity, sizeof(char))) {
            lw_ocr_free(handle_);
            handle_ = NULL;
            throw std::runtime_error("unable to query OCR output capacities");
        }
        lines_.resize(info_.max_line_capacity);
        text_.resize(static_cast<size_t>(info_.max_text_capacity));
    }

    ~OcrEngine() { lw_ocr_free(handle_); }

    OcrOutput Run(const BgrImage& image) {
        std::lock_guard<std::mutex> guard(mutex_);
        lw_ocr_result result;
        lw_error error;
        lw_ocr_result_init(&result);
        lw_error_init(&error);
        const lw_status status = lw_ocr_run_bgr_u8(
            handle_, image.pixels.empty() ? NULL : &image.pixels[0],
            static_cast<uint64_t>(image.pixels.size()), image.width, image.height,
            image.width * 3u, &lines_[0], static_cast<uint32_t>(lines_.size()),
            &text_[0], static_cast<uint64_t>(text_.size()), &result, &error);
        if (status != LW_STATUS_OK) {
            throw ApiError(422, "ocr_failed", std::string(lw_status_string(status)) +
                ": " + error.message);
        }
        if (result.line_count > lines_.size() ||
            result.required_text_capacity > text_.size()) {
            throw std::runtime_error("native OCR returned invalid output capacities");
        }
        OcrOutput output;
        output.detected_count = result.detected_count;
        output.lines.assign(lines_.begin(), lines_.begin() + result.line_count);
        output.text.assign(text_.begin(),
            text_.begin() + static_cast<size_t>(result.required_text_capacity));
        return output;
    }

private:
    OcrEngine(const OcrEngine&);
    OcrEngine& operator=(const OcrEngine&);
    lw_ocr* handle_;
    lw_ocr_info info_;
    std::vector<lw_ocr_line> lines_;
    std::vector<char> text_;
    std::mutex mutex_;
};

std::string BuildOcrJson(const std::string& request_id, const BgrImage& image,
    const OcrOutput& output, double elapsed_ms) {
    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(9) << "{\"ok\":true,\"api_version\":1,"
         << "\"request_id\":\"" << JsonEscape(request_id) << "\","
         << "\"image_width\":" << image.width << ",\"image_height\":"
         << image.height << ",\"detected_count\":" << output.detected_count
         << ",\"result\":[";
    for (size_t index = 0u; index < output.lines.size(); ++index) {
        const lw_ocr_line& line = output.lines[index];
        if (line.text_offset > output.text.size() ||
            line.text_length > output.text.size() - static_cast<size_t>(line.text_offset)) {
            throw std::runtime_error("native OCR returned an invalid text range");
        }
        const std::string text = output.text.substr(
            static_cast<size_t>(line.text_offset), static_cast<size_t>(line.text_length));
        if (index != 0u) json << ',';
        json << "{\"text\":\"" << JsonEscape(text) << "\","
             << "\"score\":" << line.recognition_score << ','
             << "\"det_score\":" << line.box.score << ','
             << "\"cls_label\":" << line.classification_label << ','
             << "\"cls_score\":" << line.classification_score << ','
             << "\"rotation\":" << line.applied_rotation_degrees << ','
             << "\"x1\":" << line.box.x1 << ",\"y1\":" << line.box.y1 << ','
             << "\"x2\":" << line.box.x2 << ",\"y2\":" << line.box.y2 << ','
             << "\"x3\":" << line.box.x3 << ",\"y3\":" << line.box.y3 << ','
             << "\"x4\":" << line.box.x4 << ",\"y4\":" << line.box.y4 << '}';
    }
    json << "],\"timing_ms\":{\"server_total\":" << elapsed_ms << "}}";
    return json.str();
}

void SetCommonHeaders(httplib::Response& response, const std::string& request_id) {
    response.set_header("Cache-Control", "no-store");
    response.set_header("X-Content-Type-Options", "nosniff");
    response.set_header("X-API-Version", "1");
    response.set_header("X-Request-ID", request_id);
}

void SetJson(httplib::Response& response, int status, const std::string& body,
    const std::string& request_id) {
    response.status = status;
    SetCommonHeaders(response, request_id);
    response.set_content(body, "application/json; charset=utf-8");
}

void SetError(httplib::Response& response, int status, const std::string& code,
    const std::string& message, const std::string& request_id) {
    std::ostringstream json;
    json << "{\"ok\":false,\"api_version\":1,\"request_id\":\""
         << JsonEscape(request_id) << "\",\"error_code\":\""
         << JsonEscape(code) << "\",\"error\":\""
         << JsonEscape(message) << "\"}";
    SetJson(response, status, json.str(), request_id);
}

int ParsePort(const std::string& value) {
    const uint32_t parsed = ParsePositiveU32(value, "port");
    if (parsed > 65535u) throw std::runtime_error("port must be between 1 and 65535");
    return static_cast<int>(parsed);
}

Config ParseArguments(const std::vector<std::string>& arguments) {
    Config config;
    const std::string executable_directory = ExecutableDirectory(
        arguments.empty() ? NULL : arguments[0].c_str());
    const std::string package_root = ParentPath(executable_directory);
    config.models = JoinPath(package_root, "models");
    config.web_root = JoinPath(package_root, "www");
    for (size_t index = 1u; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--no-cls") {
            config.use_classifier = false;
        } else if (argument == "--host" || argument == "--port" ||
                   argument == "--models" || argument == "--www" ||
                   argument == "--ocr-workers" || argument == "--rec-max-width") {
            if (index + 1u >= arguments.size()) {
                throw std::runtime_error("missing value after " + argument);
            }
            const std::string value = arguments[++index];
            if (argument == "--host") config.host = value;
            else if (argument == "--port") config.port = ParsePort(value);
            else if (argument == "--models") config.models = value;
            else if (argument == "--www") config.web_root = value;
            else if (argument == "--ocr-workers") {
                config.ocr_workers = ParsePositiveU32(value, "ocr-workers");
                if (config.ocr_workers > 16u)
                    throw std::runtime_error("ocr-workers must be between 1 and 16");
            } else {
                config.rec_max_width = ParsePositiveU32(value, "rec-max-width");
                if (config.rec_max_width != 192u && config.rec_max_width != 320u &&
                    config.rec_max_width != 480u && config.rec_max_width != 640u &&
                    config.rec_max_width != 960u) {
                    throw std::runtime_error(
                        "rec-max-width must be one of 192, 320, 480, 640, or 960");
                }
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: lw.PPOCR.C.HttpServer [--host ADDRESS] [--port PORT] "
                         "[--models DIRECTORY] [--www DIRECTORY] [--ocr-workers COUNT] "
                         "[--rec-max-width WIDTH] [--no-cls]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (config.host.empty()) throw std::runtime_error("host must not be empty");
    return config;
}

int Run(const std::vector<std::string>& arguments) {
    const Config config = ParseArguments(arguments);
    OcrEngine engine(config);
    const std::string index_html = ReadFileUtf8(
        JoinPath(config.web_root, "index.html"), kMaxWebPageBytes);
    httplib::Server server;
    server.set_read_timeout(30, 0);
    server.set_write_timeout(30, 0);
    server.set_payload_max_length(sizeof(void*) == 4u ?
        10u * 1024u * 1024u : 50u * 1024u * 1024u);

    server.Get("/", [&index_html](const httplib::Request&, httplib::Response& response) {
        const std::string request_id = MakeRequestId();
        SetCommonHeaders(response, request_id);
        response.set_content(index_html, "text/html; charset=utf-8");
    });
    server.Get("/index.html", [&index_html](const httplib::Request&, httplib::Response& response) {
        const std::string request_id = MakeRequestId();
        SetCommonHeaders(response, request_id);
        response.set_content(index_html, "text/html; charset=utf-8");
    });
    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        const std::string request_id = MakeRequestId();
        SetJson(response, 200,
            "{\"ok\":true,\"api_version\":1,\"request_id\":\"" +
            JsonEscape(request_id) + "\",\"status\":\"ready\"}", request_id);
    });
    server.Post("/api/ocr", [&engine](const httplib::Request& request,
                                      httplib::Response& response) {
        const std::string request_id = MakeRequestId();
        const std::chrono::steady_clock::time_point started =
            std::chrono::steady_clock::now();
        try {
            const BgrImage image = ReadRequestImage(request);
            const OcrOutput output = engine.Run(image);
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            SetJson(response, 200,
                BuildOcrJson(request_id, image, output, elapsed_ms), request_id);
            std::ostringstream log;
            log << "request_id=" << request_id << " status=200 duration_ms="
                << std::fixed << std::setprecision(1) << elapsed_ms
                << " lines=" << output.lines.size();
            Log(log.str());
        } catch (const ApiError& error) {
            SetError(response, error.status(), error.code(), error.what(), request_id);
            Log("request_id=" + request_id + " status=" +
                std::to_string(error.status()) + " error_code=" + error.code());
        } catch (const std::exception& error) {
            SetError(response, 500, "internal_error", "server internal error", request_id);
            Log("request_id=" + request_id + " status=500 detail=" + error.what());
        }
    });
    server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (response.has_header("X-API-Version")) return;
        const std::string request_id = MakeRequestId();
        const int status = response.status == 413 ? 413 :
            (response.status == 405 ? 405 : 404);
        const std::string code = status == 413 ? "payload_too_large" :
            (status == 405 ? "method_not_allowed" : "not_found");
        const std::string message = status == 413 ? "request body is too large" :
            (status == 405 ? "method is not allowed" : "path does not exist");
        SetError(response, status, code, message, request_id);
    });
    server.set_exception_handler([](const httplib::Request&, httplib::Response& response,
                                    std::exception_ptr) {
        const std::string request_id = MakeRequestId();
        SetError(response, 500, "internal_error", "server internal error", request_id);
    });

    std::ostringstream startup;
    startup << "lw.PPOCR.C HTTP OCR Demo\n"
            << "HTTP: http://" << config.host << ':' << config.port << "/\n"
            << "models: " << config.models << "\n"
            << "www: " << config.web_root << "\n"
            << "OCR line workers: " << config.ocr_workers << "\n"
            << "REC maximum width: " << config.rec_max_width << "\n"
            << "HTTP workers: " << LW_HTTP_THREAD_POOL_COUNT;
    Log(startup.str());
    const bool listening = server.listen(config.host.c_str(), config.port);
    if (!listening) {
        std::cerr << "unable to listen on " << config.host << ':' << config.port << std::endl;
        return 2;
    }
    return 0;
}

}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);
    try {
        std::vector<std::string> arguments;
        for (int index = 0; index < argc; ++index) {
            arguments.push_back(WideToUtf8(argv[index]));
        }
        return Run(arguments);
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
}
#else
int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments;
        for (int index = 0; index < argc; ++index) arguments.push_back(argv[index]);
        return Run(arguments);
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
}
#endif
