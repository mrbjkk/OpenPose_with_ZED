#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
#include <typeinfo>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Display
DEFINE_bool(no_display,                 false,
        "Enable to disable the visual display.");

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
using namespace sl;
using namespace std;
cv::Mat slMat2cvMat(Mat& input);

DEFINE_string(image_dir, "examples/media/", "message");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
        // datum.cvOutputData: rendered frame with pose or heatmaps
        // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr->at(0)->cvOutputData);
            cv::waitKey(0);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    // Example: How to use the pose keypoints
    if (datumsPtr != nullptr && !datumsPtr->empty())
    {
        op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
        op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
        op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
        op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}


void configureWrapper(op::Wrapper& opWrapper)
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
    const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
    // Face and hand detectors
    const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
    const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
    // Enabling Google Logging
    const bool enableGoogleLogging = true;

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
    // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
    // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
    if (FLAGS_disable_multi_thread)
        opWrapper.disableMultiThreading();

}


int main(int argc, char **argv){

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    op::log("Starting OpenPose demo...", op::Priority::High);
    const auto opTimer = op::getTimerInit();

    // Configuring OpenPose
    op::log("Configuring OpenPose...", op::Priority::High);
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    configureWrapper(opWrapper);

    // Starting OpenPose
    op::log("Starting thread(s)...", op::Priority::High);
    opWrapper.start();

    // Configure Zed SDK
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_VGA;
    init_params.camera_fps = 60;

    //Open the camera
    sl::ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS){
        std::cout<< toString(err)<<std::endl;
        exit(-1);
    }


    // Set runtime parameters after opening the camera
    sl::RuntimeParameters runtime_param;
    runtime_param.sensing_mode = SENSING_MODE_STANDARD;

    Resolution image_size = zed.getResolution();
    int new_width = image_size.width;
    int new_height = image_size.height;

    sl::Mat depth_image(zed.getResolution(), MAT_TYPE_8U_C4);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image);
    sl::Mat point_cloud;

    // frame counter
    int fc = 0;

    char key = ' ';
    while( key != 'q' ) {
        if(zed.grab(runtime_param) == SUCCESS) {
            // create variable to retrieve image from zed camera
            sl::Mat zed_image(new_width, new_height, MAT_TYPE_8U_C4);

            // retrieve the image
            zed.retrieveImage(zed_image, VIEW_LEFT);

            // convert sl::Mat to cv::Mat. 
            // Note, image_ocv has 4-channel(see the slMat2cvMat function)
            auto image_ocv = slMat2cvMat(zed_image);

            // show the original window
            cv::imshow("origin", image_ocv);

            // display depth
            zed.retrieveImage(depth_image, VIEW_DEPTH);
            cv::imshow("Depth", depth_image_ocv);

            // create a cv::Mat variable
            cv::Mat image_ocv_RGB;

            // convert 4-channel to 3-channel, namely RGBA to RGB
            cv::cvtColor(image_ocv, image_ocv_RGB, CV_RGBA2RGB);

            // process the 3-channel image by openpose
            auto datumProcessed = opWrapper.emplaceAndPop(image_ocv_RGB);

            // print the key points
            printKeypoints(datumProcessed);

            // Display the video
            cv::imshow("image", datumProcessed->at(0)->cvOutputData);
            key = cv::waitKey(10);
            fc++;
            // std::cout<<"frame counter:"<<fc<<std::endl;
        }
        else {
            key = cv::waitKey(20);
        }
    }

    // Measuring total time
    op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);
    zed.close();
    return EXIT_SUCCESS;
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
