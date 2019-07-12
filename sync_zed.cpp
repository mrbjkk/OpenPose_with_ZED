// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
// ZED SDK 
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <typeinfo>
using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
// This worker will just read and return all the basic image file formats in a directory
class WUserInput : public op::WorkerProducer<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
    public:
        WUserInput() : mParamReader(std::make_shared<op::CameraParameterReader>())
        {
            init_params.camera_resolution = RESOLUTION_VGA;
            init_params.camera_fps = 60;
            sl::ERROR_CODE err = zed.open(init_params);
            runtime_param.sensing_mode = SENSING_MODE_STANDARD;
            sl::Resolution image_size = zed.getResolution();
            new_width = image_size.width;
            new_height = image_size.height;
            mParamReader->readParameters("/home/yurik/Pictures/ZED_calibration/leftandright_califolder/");
            mIntrinsics = mParamReader->getCameraIntrinsics();
            mExtrinsics = mParamReader->getCameraExtrinsics();
            mMatrices = mParamReader->getCameraMatrices();
        }

        void initializationOnThread() {}

        cv::Mat getFrame(size_t camera_serial)
        {
            if (camera_serial == 0)
            {
                if(zed.grab(runtime_param) == SUCCESS)
                {
                    sl::Mat zed_imagel(new_width, new_height, MAT_TYPE_8U_C4);
                    zed.retrieveImage(zed_imagel, VIEW_LEFT);
                    auto image_ocvl = slMat2cvMat(zed_imagel);
                    cv::Mat image_ocv_RGBl;
                    cv::cvtColor(image_ocvl, image_ocv_RGBl, CV_RGBA2RGB);
                    return image_ocv_RGBl;
                }
            }
            else if(camera_serial == 1)
            {
                if(zed.grab(runtime_param) == SUCCESS)
                {
                    sl::Mat zed_imager(new_width, new_height, MAT_TYPE_8U_C4);
                    zed.retrieveImage(zed_imager, VIEW_RIGHT);
                    auto image_ocvr = slMat2cvMat(zed_imager);
                    cv::Mat image_ocv_RGBr;
                    cv::cvtColor(image_ocvr ,image_ocv_RGBr, CV_RGBA2RGB);
                    return image_ocv_RGBr;
                }
            }
        }

        std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> workProducer()
        {
            try
            {
                // this block processes one frame
//                wpc++;
//                std::cout<<"workProducer has been called "<<wpc<<" times"<<std::endl;
//                auto Datums = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
//                Datums->emplace_back();
//                // declare datumsPtr
//                auto& datumsPtr = Datums->back();
//                // value datumsPtr
//                datumsPtr = std::make_shared<op::Datum>();
//                datumsPtr->cvInputData = getFrame(0);
//                if(wpc % 2 == 0)
//                {
//                    datumsPtr->cvInputData = getFrame(0);
//                }
//                else
//                {
//                    datumsPtr->cvInputData = getFrame(1);
//                }
//                datumsPtr->cvOutputData = datumsPtr->cvInputData;
//                datumsPtr->cameraMatrix = mMatrices[0];
//                return Datums;

                wpc++;
                std::cout<<"workProducer has been called "<<wpc<<" times"<<std::endl;
                if (mQueuedElements.empty())
                {
                    for (auto i = 0; i < 2; i++)
                    {
                        auto Datums = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
                        Datums->emplace_back();
                        auto& datumsPtr = Datums->back();
                        datumsPtr = std::make_shared<op::Datum>();

                        //Fill datum
                        datumsPtr->cvInputData = getFrame(i);
                        datumsPtr->cvOutputData = datumsPtr->cvInputData;
                        datumsPtr->subId = i;
                        datumsPtr->subIdMax = 1;
                        datumsPtr->cameraMatrix = mMatrices[i];

                        if (datumsPtr->cvInputData.empty())
                        {
                            std::cout<<"if has been called"<<std::endl;
                            this->stop();
                            return nullptr;
                        }
                        mQueuedElements.push(Datums);
                        std::cout<<"queue size in for: "<<mQueuedElements.size()<<std::endl;
                    }
                }
                std::cout<<"queue size after for: "<<mQueuedElements.size()<<std::endl;
                auto ret = mQueuedElements.front();
                std::cout<<"queue size after front: "<<mQueuedElements.size()<<std::endl;
                mQueuedElements.pop();
                std::cout<<"queue size after pop: "<<mQueuedElements.size()<<std::endl;
                return ret;
            }
            catch (const std::exception& e)
            {
                this->stop();
                op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            }
        }

    private:
        int wpc = 0;
        int fc = 0;
        sl::Camera zed;
        sl::InitParameters init_params;
        sl::RuntimeParameters runtime_param;
        int new_width;
        int new_height;
        std::shared_ptr<op::CameraParameterReader> mParamReader;
        std::vector<cv::Mat> mIntrinsics;
        std::vector<cv::Mat> mExtrinsics;
        std::vector<cv::Mat> mMatrices;
        std::vector<std::shared_ptr<op::WebcamReader>> mCams;
        std::queue<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>> mQueuedElements;
        std::vector<uint32_t> cam_ids;
};

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
//        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        const auto multipleView = (FLAGS_3d);
//        const auto multipleView = false;
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Initializing the user custom classes
        // Frames producer (e.g., video, webcam, ...)
        auto wUserInput = std::make_shared<WUserInput>();
        // Add custom processing
        const auto workerInputOnNewThread = true;
        opWrapper.setWorker(op::WorkerType::Input, wUserInput, workerInputOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
                FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
                poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
                FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
                (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
                FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
                op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
                (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
                (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
                FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
                FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
                FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
                FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapper.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapper.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // OpenPose wrapper
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper;
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}

cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
