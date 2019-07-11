#include <iostream>
#include <dirent.h>
#include <boost/algorithm/string/replace.hpp>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"


#include "LPRegistrar.h"

using namespace std;
using namespace cv ;
using namespace Registrar;

//////////////////////////////////////////////
enum input_t{VIDEO_IN, IMAGE_IN, CAMERA_IN};
int true_rate ;
int false_rate ;
Mat prev_frame ;
vector<Point2f> prev_features ;
vector<Point2f> next_features ;
//////////////////////////////////////////////
Mat extractEdges(Mat frame);

void extract_feature(Mat &src, vector<float> &feat_vec);

void processFrame(const Mat &frame);

bool start_video_input(const String &fullFileName);

bool start_camera_input();

void start_input(String full_file_name, input_t input_type);

bool start_image_input(String full_file_name);

Mat get_input(input_t input_type);

Mat get_video_input();

Mat get_image_input();

void image_preprocessing(Mat &frame);

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
void image_sharpening(const Mat &frame);
#pragma clang diagnostic pop

void process_input_stream(const input_t &input_type, const String &fullFileName);

void decrease_illumination_noise(Mat &frame);

void blur_image(Mat &frame);

int pyramiding_down(Mat &frame);

Mat extract_license_plate(Mat mat, Mat src, int pyramidRatio, int x_offset, int y_offset);

Mat post_process_lp(Mat mat);

void register_lp(const string &plate_first, const string &plate_left, const string &letter, const string &region);

string predict(const Mat &feat);

Mat generate_feature_mat(vector<Mat> &splitted_characters, int i);

void fill_lp_details(String &plate_first, String &letter, String &plate_left, String &region, int index,
                     const string &label);

int read_lp(vector<Mat> &splitted_characters, String &car_license, String &plate_first, String &letter, String &plate_left,
        String &region);

void cut_lp_search_region(Mat &frame, int &cut_x_offset, int &cut_y_offset);

void calculate_lp_histogram(Mat &LP, int &hist_h, vector<int> &hist_data, int &max_sum, double &hist_mean);

void draw_lp_histogram(const Mat &LP, int hist_h, const Mat &histImage, vector<int> &hist_data, int hist_h_ratio,
                       float split_thresh_ratio, double char_split_threshold);

void split_lp_characters(const Mat &LP, vector<int> &hist_data, int hist_h_ratio, double char_split_threshold,
                         vector<Mat> &splitted_characters);

void write_and_show_characters(vector<Mat> &splitted_characters);

//////////////////////////////////////////////
bool has_more_input = false;
VideoCapture video;
string input_image_address;

int frame_number;
int last_seen_frame = -1;
int seen_frame_count;
int feat_row, feat_col, max_feat_size;

int main(int argc, char** argv) {

//    cout << "loading data..." << endl;
    string dataDirName = "/home/ahmad/Programs/Programming/CarLicensePlateDetection/Dataset" ;

    input_t input_type = VIDEO_IN;

    if(argc > 1){
        int input_index = 0;
        if(strcmp(argv[1], "--image") == 0) {
            input_type = IMAGE_IN;
            input_index ++;
        }
        if(strcmp(argv[1],"-c")==0 || argc == input_index+3){
            if(strcmp(argv[input_index+1] , "-d") == 0) {
                dataDirName = argv[input_index + 2];
                input_type = input_type == IMAGE_IN ? input_type : VIDEO_IN;
            }
            else if(strcmp(argv[input_index+1] , "-c") == 0) {
                input_type = CAMERA_IN;
            }
        } else {
            cout << "Wrong number of inputs" << endl;
            return -1;
        }
    }

    DIR * dir;
    struct dirent * ent;
    if (input_type != CAMERA_IN && (dir = opendir(dataDirName.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name, "..") == 0)
                continue;
            cout << "***********************************" << endl;
            cout << "Processing file: " << ent->d_name << endl;
            String fullFileName = dataDirName + "/" + ent->d_name;

            process_input_stream(input_type, fullFileName);
        }
        closedir(dir);
    }else if(input_type == CAMERA_IN){
        process_input_stream(input_type, "");

    } else {
        /* could not open directory */
        perror ("could not open directory");
        return EXIT_FAILURE;
    }

    cout << "process has been finished." << endl ;
    cout << "Precision: " << (double)true_rate/(true_rate + false_rate) << endl ;
    cout << "Recall: " << (double)(true_rate + false_rate)/frame_number << endl ;
    return 0;
}

void process_input_stream(const input_t &input_type, const String &fullFileName) {
//    cout << "start video ..." << endl ;
    start_input(fullFileName, input_type);

    while (has_more_input) {
//        cout << "has more input" << endl;
        Mat frame = get_input(input_type);
        if (!frame.empty())
            processFrame(frame);
        frame_number ++;
    }
}

Mat get_input(input_t input_type) {
    if(input_type == VIDEO_IN || input_type == CAMERA_IN)
        return get_video_input();
    else
        return get_image_input();
}

Mat get_image_input() {
    has_more_input = false;
    return imread(input_image_address);
}

Mat get_video_input() {
    Mat frame;
    video >> frame ;
    has_more_input = !frame.empty();
    return frame;
}

void start_input(String full_file_name, input_t input_type) {
    if(input_type == VIDEO_IN)
        has_more_input = start_video_input(full_file_name);
    else if(input_type == CAMERA_IN)
        has_more_input = start_camera_input();
    else
        has_more_input = start_image_input(full_file_name);
}

bool start_image_input(String full_file_name) {
    input_image_address = full_file_name;
    return true;
}

bool start_video_input(const String &fullFileName) {
    cout << "trying to open video file '" << fullFileName << "' ..." << endl ;
    video = VideoCapture(fullFileName) ;
    return video.isOpened();
}

bool start_camera_input() {
    cout << "trying to open camera ..." << endl ;
    video = VideoCapture("rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=1.sdp?") ;
    return video.isOpened();
}

void processFrame(const Mat &input_frame) {
    Mat gray_frame;
    Rect cutting_rect(input_frame.cols/4,0,input_frame.cols*2/3, input_frame.rows);
    cvtColor(input_frame, gray_frame, CV_BGR2GRAY);
    pyrDown(gray_frame,gray_frame);
    pyrDown(gray_frame,gray_frame);
    pyrDown(gray_frame,gray_frame);
    Rect rect(gray_frame.cols/4,0,gray_frame.cols*2/3, gray_frame.rows);
    gray_frame = gray_frame(rect);
//    cv::goodFeaturesToTrack(gray_frame, // the image
//                            next_features,   // the output detected features
//                            500,  // the maximum number of features
//                            0.5,     // quality level
//                            100     // min distance between two features
//    );
//
    Mat status_mat;

    Mat error;

    if(!prev_frame.empty()){
        calcOpticalFlowFarneback(
                prev_frame,
                gray_frame,
                status_mat,
                0.5,
                1,
                7,
                5,
                5,
                1,
                0
        );

//        imshow("current_frame", gray_frame);
        Mat channels[2];
        split(status_mat,channels);
        Mat map = channels[0] + channels[1];
        threshold(map, map, 0.5, 256, THRESH_BINARY);
        map.convertTo(map, CV_8UC1);
//        medianBlur(map, map, 7);
//        medianBlur(map, map, 11);

        Scalar scalar = sum(status_mat);
        int thresh = 3000;
        int last_seen_thresh = 5 * 18;
        if(scalar[0] > thresh){
            last_seen_frame = frame_number;
        }
        if(scalar[0] > thresh || (frame_number - last_seen_frame) < last_seen_thresh){
//            cout << scalar[0] << endl;
//            imshow("car", gray_frame);
            string out_file_name = string("/tmp/carDetecteds/") + to_string(frame_number) + ".png";
            cout << "frame writed to " << out_file_name << ", seen frames: " << seen_frame_count << endl;
            imwrite(out_file_name, input_frame(cutting_rect));
            seen_frame_count ++;
        }
//        imshow("src", input_frame);
//        imshow("map", map);
//        waitKey(1);
    }

    prev_frame = gray_frame.clone();
    prev_features = next_features;
    return;


}

