3#include "Person_Detect.h"
#include <unistd.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <string>
#include <cstdlib>
#include <mutex>
#include <glob.h>
#include <dirent.h>
#include <stdio.h>


Person_Detect::Person_Detect()
    : m_camera_ready(false), m_image(nullptr), m_image_reader(nullptr), m_native_camera(nullptr){}

Person_Detect::~Person_Detect(){
    JNIEnv *env;
    java_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    env->DeleteGlobalRef(calling_activity_obj);
    calling_activity_obj = nullptr;

    // ACameraCaptureSession_stopRepeating(m_capture_session);
    if (m_native_camera != nullptr) {
        delete m_native_camera;
        m_native_camera = nullptr;
    }

    // make sure we don't leak native windows
    if (m_native_window != nullptr) {
        ANativeWindow_release(m_native_window);
        m_native_window = nullptr;
    }

    if (m_image_reader != nullptr) {
        delete (m_image_reader);
        m_image_reader = nullptr;
    }
}

void Person_Detect::OnCreate() {

    qc = new Qcsnpe(model_path, 2, output_layers);

}

void Person_Detect::OnPause() {}
void Person_Detect::OnDestroy() {}

void Person_Detect::SetNativeWindow(ANativeWindow* native_window) {
    // Save native window
    m_native_window = native_window;
}

void Person_Detect::SetUpCamera() {

    m_native_camera = new Native_Camera(m_selected_camera_type);
    m_native_camera->MatchCaptureSizeRequest(&m_view,
                                             ANativeWindow_getWidth(m_native_window),
                                             ANativeWindow_getHeight(m_native_window));

    LOGI("______________mview %d\t %d\n", m_view.width, m_view.height);
    LOGI("______________mview %d\t %d\n", ANativeWindow_getWidth(m_native_window),ANativeWindow_getHeight(m_native_window));
    ASSERT(m_view.width && m_view.height, "Could not find supportable resolution");

    ANativeWindow_setBuffersGeometry(m_native_window, m_view.width, m_view.height,
                                     WINDOW_FORMAT_RGBX_8888);
    m_image_reader = new Image_Reader(&m_view, AIMAGE_FORMAT_YUV_420_888);

    m_image_reader->SetPresentRotation(m_native_camera->GetOrientation());
    ANativeWindow* image_reader_window = m_image_reader->GetNativeWindow();
    m_camera_ready = m_native_camera->CreateCaptureSession(image_reader_window);
}

//std::string class_name_path = "/storage/emulated/0/appData/models/classes.txt";
std::string class_name_path = "/sdcard/Documents/classes.txt";
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs(class_name_path);
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}
std::vector<std::string> class_list = load_class_list();

void Person_Detect::CameraLoop() {
    bool buffer_printout = false;
    video_writer.open("/sdcard/Documents/Person_Detect_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size(640, 480), true);

    while (1) {
        if (m_camera_thread_stopped) { break; }
        if (!m_camera_ready || !m_image_reader) { continue; }
        //reading the image from ndk reader
        m_image = m_image_reader->GetNextImage();
        if (m_image == nullptr) { continue; }

        ANativeWindow_acquire(m_native_window);
        ANativeWindow_Buffer buffer;
        if (ANativeWindow_lock(m_native_window, &buffer, nullptr) < 0) {
            m_image_reader->DeleteImage(m_image);
            m_image = nullptr;
            continue;
        }
        if (false == buffer_printout) {
            buffer_printout = true;
            LOGI("/// H-W-S-F: %d, %d, %d, %d", buffer.height, buffer.width, buffer.stride,
                 buffer.format);
        }

        //display the image
        m_image_reader->DisplayImage(&buffer, m_image);

        //converting the ndk image into opencv format
        img_mat = cv::Mat(buffer.height, buffer.stride, CV_8UC4, buffer.bits);
        //cv::imwrite("/storage/emulated/0/appData/models/input.jpg",img_mat);
        cv::Mat src_img = img_mat.clone();

        bgr_img = cv::Mat(img_mat.rows, img_mat.cols, CV_8UC3);

        cv::cvtColor(img_mat, bgr_img, cv::COLOR_RGBA2BGR);
        // bgr_img is normal image
        //cv::imwrite("/storage/emulated/0/appData/models/inp.jpg",bgr_img);

        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        std::vector<Detection> output;
        cv::Mat res_img = cv::Mat(640, 640, CV_8UC3);
        cv::resize(bgr_img, res_img, cv::Size(640, 640));
        // res_img is pre-processed image we are passing it for inference
        //cv::imwrite("/storage/emulated/0/appData/models/inp.jpg",res_img);

        pred_out = qc->predict(res_img);
        std::vector<float> out_arr = pred_out["output"];
        std::vector<cv::Mat> outputs;

        outputs.emplace_back(cv::Mat(out_arr));
        float x_factor = res_img.cols / INPUT_WIDTH;
        float y_factor = res_img.rows / INPUT_HEIGHT;

        //float *data = (float *)outputs[0].data;
        float *data = (float *)out_arr.data();
        //const int dimensions = 85;
        const int dimensions = 6;
        const int rows = 25200;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;


        for (int i = 0; i < rows; ++i) {

            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {

                float * classes_scores = data + 5;
                cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > SCORE_THRESHOLD) {

                    confidences.push_back(confidence);

                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }

            }
            data += 6;

        }

        //std::cout << class_ids.size() << std::endl;

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        for (int i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(result);
        }
        int detections = output.size();

        LOGI("%d", detections);
        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];

            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(img_mat, box, color, 3);
            cv::rectangle(img_mat, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        }
        cv::imwrite("/storage/emulated/0/appData/models/Person_Detect_bgr.jpg",bgr_img);
        cv::resize(img_mat, out_img, cv::Size(640, 480));
        video_writer.write(out_img);
        cv::imwrite("/storage/emulated/0/appData/models/Person_Detect_image.jpg",out_img);

        pred_out.clear();
        ANativeWindow_unlockAndPost(m_native_window);
        ANativeWindow_release(m_native_window);
    }
    video_writer.release();

}

