#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;

enum class ScannerProfile {
    Generic,
    Ibml
};

struct Config {
    double min_score = 0.55;
    int max_retries = 1;
    unsigned int workers = std::max(1u, std::thread::hardware_concurrency() > 1
                                            ? std::thread::hardware_concurrency() - 1
                                            : 1u);
    std::string language = "eng";
    ScannerProfile scanner_profile = ScannerProfile::Generic;

    bool enable_api = false;
    std::string api_host = "0.0.0.0";
    int api_port = 8000;

    fs::path input_dir = "scan_input";
    fs::path output_dir = "scan_output";
    fs::path reject_dir = "scan_reject";
};

struct Job {
    fs::path path;
    int retries = 0;
    std::vector<std::string> history;
};

struct Stats {
    int processed = 0;
    int ok = 0;
    int retry = 0;
    int reject = 0;
    std::vector<std::string> last_files;
};

class JobQueue {
public:
    void push(Job job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(job));
        }
        cv_.notify_one();
    }

    std::optional<Job> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });

        if (stop_ && queue_.empty()) {
            return std::nullopt;
        }

        Job job = std::move(queue_.front());
        queue_.pop();
        return job;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<Job> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

class OcrGatekeeper {
public:
    explicit OcrGatekeeper(Config cfg) : config_(std::move(cfg)) {
        fs::create_directories(config_.output_dir);
        fs::create_directories(config_.reject_dir);
    }

    int enqueue_scan_input() {
        int enqueued = 0;
        for (const auto& entry : fs::directory_iterator(config_.input_dir)) {
            if (!entry.is_regular_file() || !is_supported_extension(entry.path())) {
                continue;
            }

            queue_.push(Job{entry.path()});
            pending_.fetch_add(1);
            ++enqueued;
        }
        return enqueued;
    }

    void start_workers() {
        for (unsigned int i = 0; i < config_.workers; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void run_until_idle() {
        while (pending_.load() > 0 || !queue_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void shutdown() {
        queue_.stop();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    std::string status_json() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        std::ostringstream out;
        out << "{\"processed\":" << stats_.processed
            << ",\"ok\":" << stats_.ok
            << ",\"retry\":" << stats_.retry
            << ",\"reject\":" << stats_.reject
            << ",\"last_files\":[";

        for (size_t i = 0; i < stats_.last_files.size(); ++i) {
            if (i > 0) out << ',';
            out << '"' << escape_json(stats_.last_files[i]) << '"';
        }
        out << "]}";
        return out.str();
    }

    std::string dashboard_html() {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        std::ostringstream files;
        for (const auto& f : stats_.last_files) {
            files << f << "<br>";
        }

        std::ostringstream html;
        html << "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
             << "<title>Scan Pipeline Dashboard</title>"
             << "<meta http-equiv='refresh' content='2'>"
             << "<style>body{font-family:Arial;background:#111;color:#eee;margin:24px;}"
             << ".card{padding:16px;margin:8px 0;background:#222;border-radius:8px;}"
             << "button{padding:10px 14px;background:#0a84ff;color:#fff;border:0;border-radius:6px;cursor:pointer;}"
             << "</style></head><body>"
             << "<h1>📄 OCR Gatekeeper Dashboard (C++)</h1>"
             << "<form method='post' action='/scan'><button type='submit'>Scan starten</button></form>"
             << "<div class='card'>Processed: " << stats_.processed << "</div>"
             << "<div class='card'>OK: " << stats_.ok << "</div>"
             << "<div class='card'>Retry: " << stats_.retry << "</div>"
             << "<div class='card'>Reject: " << stats_.reject << "</div>"
             << "<h2>Last Files</h2><div class='card'>" << files.str() << "</div>"
             << "</body></html>";

        return html.str();
    }

private:
    static std::string escape_json(const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (char c : in) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
            }
        }
        return out;
    }

    static bool is_supported_extension(const fs::path& path) {
        static const std::vector<std::string> allowed = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        };

        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return std::find(allowed.begin(), allowed.end(), ext) != allowed.end();
    }

    static cv::Mat deskew_if_needed(const cv::Mat& binary) {
        std::vector<cv::Point> points;
        cv::findNonZero(255 - binary, points);
        if (points.size() < 50) {
            return binary;
        }

        cv::RotatedRect box = cv::minAreaRect(points);
        double angle = box.angle;
        if (angle < -45.0) {
            angle += 90.0;
        }

        if (std::abs(angle) < 0.75) {
            return binary;
        }

        cv::Mat rotated;
        cv::Point2f center(binary.cols / 2.0f, binary.rows / 2.0f);
        cv::Mat matrix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(binary, rotated, matrix, binary.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return rotated;
    }

    static cv::Mat preprocess_generic(const cv::Mat& image, int mode) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        switch (mode) {
            case 0: {
                cv::Mat out;
                cv::equalizeHist(gray, out);
                return out;
            }
            case 1: {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
                cv::Mat out;
                clahe->apply(gray, out);
                return out;
            }
            case 2: {
                cv::Mat out;
                cv::adaptiveThreshold(gray, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv::THRESH_BINARY, 31, 10);
                return out;
            }
            default:
                return gray;
        }
    }

    static cv::Mat preprocess_ibml(const cv::Mat& image, int mode) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat denoised;
        cv::medianBlur(gray, denoised, 3);

        cv::Mat normalized;
        cv::normalize(denoised, normalized, 0, 255, cv::NORM_MINMAX);

        switch (mode) {
            case 0: {
                cv::Mat out;
                cv::adaptiveThreshold(normalized, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv::THRESH_BINARY, 41, 11);
                return deskew_if_needed(out);
            }
            case 1: {
                cv::Mat out;
                cv::threshold(normalized, out, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                cv::morphologyEx(out, out, cv::MORPH_OPEN, kernel);
                return deskew_if_needed(out);
            }
            case 2: {
                cv::Mat clahe_out;
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(normalized, clahe_out);
                cv::Mat out;
                cv::adaptiveThreshold(clahe_out, out, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                      cv::THRESH_BINARY, 35, 8);
                return deskew_if_needed(out);
            }
            default:
                return normalized;
        }
    }

    cv::Mat preprocess(const cv::Mat& image, int mode) const {
        if (config_.scanner_profile == ScannerProfile::Ibml) {
            return preprocess_ibml(image, mode);
        }
        return preprocess_generic(image, mode);
    }

    double ocr_score(const cv::Mat& image) const {
        tesseract::TessBaseAPI api;
        if (api.Init(nullptr, config_.language.c_str())) {
            return 0.0;
        }

        api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        api.SetImage(image.data, image.cols, image.rows, image.channels(), static_cast<int>(image.step));
        api.Recognize(nullptr);

        tesseract::ResultIterator* ri = api.GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

        if (!ri) {
            api.End();
            return 0.0;
        }

        double sum = 0.0;
        int count = 0;

        do {
            float conf = ri->Confidence(level);
            if (conf >= 0.0f) {
                sum += conf;
                ++count;
            }
        } while (ri->Next(level));

        api.End();
        if (count == 0) {
            return 0.0;
        }

        return (sum / static_cast<double>(count)) / 100.0;
    }

    std::vector<cv::Mat> load_pages(const fs::path& path) {
        std::vector<cv::Mat> pages;
        const std::string ext = path.extension().string();

        if (ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
            cv::imreadmulti(path.string(), pages, cv::IMREAD_COLOR);
        }

        if (pages.empty()) {
            cv::Mat single = cv::imread(path.string(), cv::IMREAD_COLOR);
            if (!single.empty()) {
                pages.push_back(single);
            }
        }

        return pages;
    }

    void process_job(Job job) {
        std::vector<cv::Mat> pages = load_pages(job.path);
        if (pages.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.reject++;
            pending_.fetch_sub(1);
            return;
        }

        double file_best_score = 0.0;
        bool wrote_output = false;

        for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
            double best_score = 0.0;
            cv::Mat best_img;

            for (int mode = 0; mode < 3; ++mode) {
                cv::Mat pre = preprocess(pages[page_idx], mode);
                double score = ocr_score(pre);

                std::ostringstream s;
                s << "p" << page_idx << ":" << mode << ':' << std::fixed << std::setprecision(2) << score;
                job.history.push_back(s.str());

                if (score > best_score) {
                    best_score = score;
                    best_img = pre;
                }
            }

            file_best_score = std::max(file_best_score, best_score);
            if (best_score >= config_.min_score) {
                fs::path out_name = job.path.stem().string() + "_p" + std::to_string(page_idx) + ".png";
                cv::imwrite((config_.output_dir / out_name).string(), best_img);
                wrote_output = true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.processed++;
            std::ostringstream rec;
            rec << job.path.filename().string() << " (" << std::fixed << std::setprecision(2) << file_best_score << ')';
            stats_.last_files.push_back(rec.str());
            if (stats_.last_files.size() > 10) {
                stats_.last_files.erase(stats_.last_files.begin());
            }
        }

        if (wrote_output) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.ok++;
            pending_.fetch_sub(1);
            return;
        }

        if (job.retries < config_.max_retries) {
            job.retries++;
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.retry++;
            }
            queue_.push(std::move(job));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.reject++;
        }
        std::error_code ec;
        fs::rename(job.path, config_.reject_dir / job.path.filename(), ec);
        pending_.fetch_sub(1);
    }

    void worker_loop() {
        while (true) {
            auto maybe_job = queue_.pop();
            if (!maybe_job.has_value()) {
                break;
            }
            process_job(std::move(*maybe_job));
        }
    }

private:
    Config config_;
    JobQueue queue_;
    Stats stats_;
    std::mutex stats_mutex_;
    std::vector<std::thread> workers_;
    std::atomic<int> pending_{0};
};

class SimpleHttpServer {
public:
    SimpleHttpServer(OcrGatekeeper& gatekeeper, Config config)
        : gatekeeper_(gatekeeper), config_(std::move(config)) {}

    void run() {
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "Could not create server socket\n";
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(config_.api_port));
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "Bind failed on port " << config_.api_port << '\n';
            close(server_fd);
            return;
        }

        if (listen(server_fd, 16) < 0) {
            std::cerr << "Listen failed\n";
            close(server_fd);
            return;
        }

        std::cout << "Dashboard/API running on http://" << config_.api_host << ':' << config_.api_port << '\n';

        while (true) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                continue;
            }
            std::thread(&SimpleHttpServer::handle_client, this, client_fd).detach();
        }
    }

private:
    static std::string make_response(const std::string& body, const std::string& content_type, int status,
                                     const std::string& status_text) {
        std::ostringstream out;
        out << "HTTP/1.1 " << status << ' ' << status_text << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << body;
        return out.str();
    }

    void handle_client(int client_fd) {
        std::string request;
        request.resize(8192);
        ssize_t bytes = recv(client_fd, request.data(), request.size(), 0);
        if (bytes <= 0) {
            close(client_fd);
            return;
        }
        request.resize(static_cast<size_t>(bytes));

        std::istringstream in(request);
        std::string method;
        std::string path;
        std::string version;
        in >> method >> path >> version;

        std::string response;
        if (method == "GET" && path == "/status") {
            response = make_response(gatekeeper_.status_json(), "application/json; charset=utf-8", 200, "OK");
        } else if (method == "POST" && path == "/scan") {
            int count = gatekeeper_.enqueue_scan_input();
            std::ostringstream body;
            body << "{\"message\":\"Jobs added\",\"count\":" << count << '}';
            response = make_response(body.str(), "application/json; charset=utf-8", 200, "OK");
        } else if (method == "GET" && path == "/") {
            response = make_response(gatekeeper_.dashboard_html(), "text/html; charset=utf-8", 200, "OK");
        } else {
            response = make_response("Not Found", "text/plain; charset=utf-8", 404, "Not Found");
        }

        send(client_fd, response.c_str(), response.size(), 0);
        close(client_fd);
    }

    OcrGatekeeper& gatekeeper_;
    Config config_;
};

static ScannerProfile parse_profile(const std::string& value) {
    if (value == "ibml" || value == "IBML") {
        return ScannerProfile::Ibml;
    }
    return ScannerProfile::Generic;
}

static bool parse_bool(const char* value) {
    if (!value) return false;
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static Config load_config_from_env() {
    Config cfg;

    if (const char* profile = std::getenv("OCR_PROFILE")) {
        cfg.scanner_profile = parse_profile(profile);
    }
    if (const char* lang = std::getenv("OCR_LANG")) {
        cfg.language = lang;
    }
    if (const char* in = std::getenv("OCR_INPUT_DIR")) {
        cfg.input_dir = in;
    }
    if (const char* out = std::getenv("OCR_OUTPUT_DIR")) {
        cfg.output_dir = out;
    }
    if (const char* rej = std::getenv("OCR_REJECT_DIR")) {
        cfg.reject_dir = rej;
    }
    if (const char* enable_api = std::getenv("OCR_ENABLE_API")) {
        cfg.enable_api = parse_bool(enable_api);
    }
    if (const char* api_port = std::getenv("OCR_API_PORT")) {
        cfg.api_port = std::max(1, std::atoi(api_port));
    }
    if (const char* api_host = std::getenv("OCR_API_HOST")) {
        cfg.api_host = api_host;
    }

    if (cfg.scanner_profile == ScannerProfile::Ibml) {
        cfg.min_score = 0.62;
        cfg.max_retries = 2;
    }

    return cfg;
}

int main() {
    Config config = load_config_from_env();

    if (!fs::exists(config.input_dir)) {
        std::cerr << "Input directory not found: " << config.input_dir << '\n';
        return 1;
    }

    OcrGatekeeper gatekeeper(config);
    gatekeeper.start_workers();

    if (config.enable_api) {
        SimpleHttpServer server(gatekeeper, config);
        server.run();
        gatekeeper.shutdown();
        return 0;
    }

    gatekeeper.enqueue_scan_input();
    gatekeeper.run_until_idle();
    gatekeeper.shutdown();
    std::cout << gatekeeper.status_json() << '\n';

    return 0;
}
