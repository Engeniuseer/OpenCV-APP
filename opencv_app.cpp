

#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h> 
using namespace cv; 


void displayimage(cv::Mat frame,cv::Mat menuimage){
    cv::Mat stackedimage;
    cv::vconcat(frame, menuimage,stackedimage);
    cv::namedWindow("CV project", WINDOW_AUTOSIZE); 
    cv::imshow("CV project", stackedimage);
}


cv::Mat gaussian_blurr(cv::Mat frame){
    cv::Mat gaussimage;
    cv::GaussianBlur(frame,gaussimage,cv::Size(7,7),3.0);
    return gaussimage;
}

cv::Mat edge_detection(cv::Mat frame){
    cv::Mat cannyimage;
    cv::Canny(frame,cannyimage,75,120);
    cv::cvtColor(cannyimage,cannyimage,cv::COLOR_GRAY2RGB);
    return cannyimage;
}

cv::Mat face_detection(cv::Mat frame,cv::CascadeClassifier faceCascade){
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    // Detect faces
    std::vector<cv::Rect> faces;
    //faceCascade.detectMultiScale(gray,faces,1.1,10,cv::Size(100, 100));
    std::vector<double> weights;
    std::vector<int> levels;
    //faceCascade.detectMultiScale(gray, faces, levels, weights, 1.1, 3, 0, cv::Size(), cv::Size(), true);
    faceCascade.detectMultiScale(gray, faces, 1.1, 10, 0);
    // Draw rectangles around faces
    for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);  // Blue color
    }
    return frame;
}

cv::Mat cartoonization(cv::Mat frame){

    //CARTOONIZATION
    int num_down=2;
    int num_bilateral=7;
    //return frame;
    
    cv::Mat img_color = frame;
    
    // Downsample
    for (int i = 0; i < num_down; ++i) {
        cv::pyrDown(img_color, img_color);
    }
    
    // Bilateral filter
    for (int i = 0; i < num_bilateral; ++i) {
        cv::Mat temp;
        cv::bilateralFilter(img_color, temp, 9, 9, 7);
        img_color=temp.clone();
    }
    
    // Upsample
    for (int i = 0; i < num_down; ++i) {
        cv::Mat temp;
        cv::pyrUp(img_color, temp);
        img_color=temp.clone();
    }
    cv::resize(img_color,img_color,frame.size());
    // Convert RGB to Gray
    cv::Mat img_gray;
    cv::cvtColor(frame, img_gray, cv::COLOR_RGB2GRAY);

    // Median Blur
    cv::Mat img_blur;
    cv::medianBlur(img_gray, img_blur, 7);

    // Adaptive Threshold
    cv::Mat img_edge;
    cv::adaptiveThreshold(img_blur, img_edge, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 2);
    cv::resize(img_edge,img_edge,frame.size());
    // Convert Gray to RGB
    cv::cvtColor(img_edge, img_edge, cv::COLOR_GRAY2RGB);
    // Bitwise AND operation
    cv::Mat result;
    cv::bitwise_and(img_color, img_edge, result);
    return result;
}




int main(int argc, char** argv) 
{ 
    cv::Mat original_image; 
    original_image = cv::imread("lena.jpg", 1); 
    if (!original_image.data) { 
        printf("No image data \n"); 
        return -1; 
    } 
    cv::Mat menuimage; 
    menuimage = cv::imread("menucv.png", 1); 
    if (!menuimage.data) { 
        printf("No image data \n"); 
        return -1; 
    }

    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade." << std::endl;
        return -1;
    }
    displayimage(original_image,menuimage);
    while(true){
        cv::Mat frame;
        frame=original_image.clone();

        int k= cv::waitKey(1);

        if (k=='1'){
            displayimage(frame,menuimage);
        }
        else if (k=='2'){
            cv::Mat result=gaussian_blurr(frame);
            displayimage(result,menuimage);
        }
        else if (k=='3'){
            cv::Mat result=cartoonization(frame);
            displayimage(result,menuimage);
        }
        else if (k=='4'){
            cv::Mat result=edge_detection(frame);
            displayimage(result,menuimage);
        }
        else if (k=='5'){
            cv::Mat result=face_detection(frame,faceCascade);
            displayimage(result,menuimage);
        }
        else if (k==27){
            return 0;
        }
    }
}
