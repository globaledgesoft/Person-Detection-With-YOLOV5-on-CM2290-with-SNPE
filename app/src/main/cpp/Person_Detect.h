
#ifndef PERSONDETECTION_CAMERAAPP_H
#define PERSONDETECTION_CAMERAAPP_H


#include <android/asset_manager.h>
#include <android/native_window.h>
#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>

#include "Image_Reader.h"
#include "Native_Camera.h"
#include "Util.h"

#include <unistd.h>
#include <time.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include "qcsnpe.hpp"
//#include "utils.hpp"
#define OUTPUT_LAYER_1 "/model.24/Concat_15"


class Person_Detect{
public:
    Person_Detect();
    ~Person_Detect();
    Person_Detect(const Person_Detect& other) = delete;
    Person_Detect& operator=(const Person_Detect& other) = delete;

    void OnCreate();
    void OnPause();
    void OnDestroy();
    void SetJavaVM(JavaVM* pjava_vm) { java_vm = pjava_vm; }
    void SetNativeWindow(ANativeWindow* native_window);

    void SetUpCamera();

    void CameraLoop();
private:
    JavaVM* java_vm;
    jobject calling_activity_obj;
    ANativeWindow* m_native_window;
    ANativeWindow_Buffer m_native_buffer;
    Native_Camera* m_native_camera;
    camera_type m_selected_camera_type = BACK_CAMERA; // Default
    ImageFormat m_view{0, 0, 0};
    Image_Reader* m_image_reader;
    AImage* m_image;

    volatile bool m_camera_ready;
    // for timing OpenCV bottlenecks
    clock_t start_t, end_t;
    // Used to detect up and down motion
    bool scan_mode;

    // OpenCV values
    cv::Mat img_mat;
    cv::Mat bgr_img;
    cv::Mat grey_img;
    cv::Mat rgb_img;

    cv::Mat out_img;
    bool m_camera_thread_stopped = false;
    Qcsnpe *qc;
    


    cv::VideoWriter video_writer;

//    std::string model_path = "/storage/emulated/0/appData/models/yolov5_person_latest.dlc";
    std::string model_path = "/sdcard/Download/Telegram/yolov5_person_latest.dlc";
    std::vector<std::string> output_layers {OUTPUT_LAYER_1};

    std::map<std::string, std::vector<float>> pred_out;
    const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0)};

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;

    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };


    };


#endif //ONETRY_CAMERAAPP_H
