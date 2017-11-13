#define USE_OPENCV

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace caffe;

const string label[21] ={"background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
                         "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};

class Detector{
private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    unsigned int num_channels_;
    cv::Mat mean_;
    
public:
    Detector(const string& model_file,
             const string& weights_file,
             const string& mean_file,
             const string& mean_value);

    std::vector<std::vector<float> > Detect(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

    void SetMean(const string& mean_file, const string& mean_value);

}; //Detector

void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (unsigned int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (unsigned int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

    Detector::Detector(const string& model_file,
                       const string& weights_file,
                       const string& mean_file = "",
                       const string& mean_value = "104,117,123"){
        Caffe::set_mode(Caffe::GPU);
        net_.reset(new Net<float>(model_file,TEST));
        net_->CopyTrainedLayersFrom(weights_file);

        Blob<float>* input_layer = net_->input_blobs()[0]; // The following three steps are using to calibrate the
        num_channels_ = input_layer->channels();           // size of input blob layer
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        SetMean(mean_file, mean_value);
    }

    std::vector<std::vector<float> > Detector::Detect(const cv::Mat &img){
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                             input_geometry_.height, input_geometry_.width);
        /* Forward dimension change to all layers. */
        net_->Reshape();

        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);

        Preprocess(img, &input_channels);

        net_->Forward();

        /* Copy the output layer to a std::vector */
        Blob<float>* result_blob = net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        const int num_det = result_blob->height();
        vector<vector<float> > detections;
        for (int k = 0; k < num_det; ++k) {
          if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
          }
          vector<float> detection(result, result + 7);
          detections.push_back(detection);
          result += 7;
        }
        return detections;
    }

    void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
      Blob<float>* input_layer = net_->input_blobs()[0];

      int width = input_layer->width();
      int height = input_layer->height();
      float* input_data = input_layer->mutable_cpu_data();
      for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
      }
    }

    void Detector::Preprocess(const cv::Mat& img,
                                std::vector<cv::Mat>* input_channels) {
      /* Convert the input image to the input image format of the network. */
      cv::Mat sample;
      if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
      else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
      else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
      else
        sample = img;

      cv::Mat sample_resized;
      if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
      else
        sample_resized = sample;

      cv::Mat sample_float;
      if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
      else
        sample_resized.convertTo(sample_float, CV_32FC1);

      cv::Mat sample_normalized;
      cv::subtract(sample_float, mean_, sample_normalized);// !

      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, *input_channels);

      CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    }

    void preLineHough(cv::Mat imageOrg,
                      vector<cv::Vec4i>* Leftline,
                      vector<cv::Vec4i>* Rightline,
                      vector<double>*   l_slope,
                      vector<double>*   r_slope,
                      vector<double>*  l_b,
                      vector<double>*  r_b) {

        cv::Mat dst,dst_HSL;
        cv::Mat s_plane,L_plane;
        int x       = imageOrg.cols * 6/32,
            y       = imageOrg.rows * 5/8,
            width   = imageOrg.cols * 22/32,
            height  = imageOrg.rows * 3/8;
        cv::cvtColor(imageOrg,dst,CV_RGB2Lab);
        cv::cvtColor(imageOrg,dst_HSL,CV_RGB2YCrCb);
        cv::Mat imageROI(dst,cv::Rect(x,y,width,height));
        cv::Mat imageROI_HSL(dst_HSL,cv::Rect(x,y,width,height));
        vector<cv::Mat> temp,temp_HSL;
        cv::split(imageROI,temp);
        cv::split(imageROI_HSL,temp_HSL);
        L_plane = temp_HSL[0];
        s_plane = temp[2];

        cv::GaussianBlur(L_plane,L_plane,cv::Size(11,11),5,5);
        cv::GaussianBlur(s_plane,s_plane,cv::Size(11,11),5,5);
        cv::threshold(L_plane,L_plane,200.0,250.0,cv::THRESH_BINARY);
        cv::threshold(s_plane,s_plane,105.0,120.0,cv::THRESH_BINARY);
        cv::Canny(s_plane,s_plane,110,95);

        s_plane = cv::max(s_plane,L_plane);

        vector<cv::Vec4i> lines;
        vector<cv::Vec4i> res;

        cv::HoughLinesP(s_plane,lines,1,CV_PI/180,80,50,5);
        vector<cv::Vec4i>::const_iterator it=lines.begin();
        while(it!=lines.end()){
            int dx = (*it)[0] - (*it)[2];
            int dy = (*it)[1] - (*it)[3];
            double angle = atan2(dy,dx) * 180 /CV_PI;
            double k_    = (double)dy/dx;
            double b     = -k_* (*it)[0] + (*it)[1];

            if(abs(angle) <= 10 || dx == 0){
                ++it;
                continue;
            }

            if ((*it)[1] > (*it)[3] + 50 || (*it)[1] < (*it)[3] - 50){
                res.push_back(*it);
                if(k_<0){
                    Leftline->push_back(*it);
                    l_slope->push_back(angle);
                    l_b->push_back(b);
                }
                else if(k_>0){
                    Rightline->push_back(*it);
                    r_slope->push_back(angle);
                    r_b->push_back(b);
                }
            }
            ++it;
         }
    }           //PrelineHough

    vector<cv::Vec4i> LineFilter(vector<cv::Vec4i> lines, vector<double> angle, vector<double> b_){

        cv::Scalar meanValue = cv::mean(angle);
        float Slope_average = meanValue.val[0];
        cv::Scalar mean_b = cv::mean(b_);
        float b_average = mean_b.val[0];
        float angle_thres = 10, d_thres = 120;

        for(unsigned int i = 0;i<lines.size();i++){
            if(abs(angle[i]-Slope_average)>angle_thres || abs(b_[i]-b_average)>d_thres)
            {
                lines.erase(lines.begin()+i);
                angle.erase(angle.begin()+i);
                b_.erase(b_.begin()+i);
                i--;
            }
        }
        return lines;
    }       //LineFilter

    vector<float> findLine(vector<cv::Vec4i> Lines, int ymin, int ymax){
        vector<cv::Point> points;
        vector<cv::Vec4i>::const_iterator it=Lines.begin();
        while(it!=Lines.end()){
            points.push_back(cv::Point((*it)[0],(*it)[1]));
            points.push_back(cv::Point((*it)[2],(*it)[3]));
            ++it;
        }
        vector<float> Line;
        cv::fitLine(points,Line,CV_DIST_L2,1,0.01,0.01);
        float k_ = Line[1]/Line[0];
        float   xup = (ymin - Line[3])/k_ + Line[2];
        float   xdown = (ymax - Line[3])/k_ + Line[2];

        Line[0] = xdown;
        Line[1] = ymax;
        Line[2] = xup;
        Line[3] = ymin;
        return Line;
    }

    void LaneDetection(cv::Mat imageOrg, vector<float>* line_L, vector<float>* line_R){
        vector<cv::Vec4i> Leftline, Rightline;
        vector<double>   l_slope, r_slope, l_b, r_b;

        preLineHough(imageOrg,&Leftline,&Rightline,&l_slope,&r_slope,&l_b,&r_b);
        int height  = imageOrg.rows * 3/8;
        Leftline = LineFilter(Leftline,l_slope,l_b);
        Rightline = LineFilter(Rightline,r_slope,r_b);
        if(Leftline.size() != 0)
        *line_L = findLine(Leftline,30,height-35);
        if(Rightline.size() != 0)
        *line_R = findLine(Rightline,30,height-45);
    }

int main()
{

    const string& model_file = "/home/cookie/ssd/ssd/deploy.prototxt";
    const string& weights_file = "/home/cookie/ssd/ssd/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
    Detector detector(model_file,weights_file);

    cv::VideoCapture capture("project_video.mp4");
        if(!capture.isOpened())
            return 1;
    cv::Mat frame;
    cv::Mat imageUncharged;

    capture.read(frame);
    vector<float> line_L,line_R;

    int x       = frame.cols * 6/32,
        y       = frame.rows * 5/8;

    while(capture.read(frame)){
        frame.copyTo(imageUncharged);
        std::vector<vector<float> > detections = detector.Detect(frame);


        /****** Traffic Lane Detection ******/
        /***
        LaneDetection(frame, &line_L, &line_R);
        if(line_L.size() != 0)
            cv::line(frame,cv::Point(x+line_L[0],y+line_L[1]),cv::Point(x+line_L[2],y+line_L[3]),cv::Scalar(40,230,60),8);
        if(line_R.size() != 0)
            cv::line(frame,cv::Point(x+line_R[0],y+line_R[1]),cv::Point(x+line_R[2],y+line_R[3]),cv::Scalar(40,230,60),8);
            ***/

        vector<cv::Point> points;
        vector<vector<cv::Point> > ptrs;
         LaneDetection(frame, &line_L, &line_R);

        if(line_L.size()!=0 && line_R.size()!=0){
            points.push_back(cv::Point((int)(x+line_L[0]),(int)(y+line_L[1])));
            points.push_back(cv::Point((int)(x+line_L[2]),(int)(y+line_L[3])));
            points.push_back(cv::Point((int)(x+line_R[2]),(int)(y+line_R[3])));
            points.push_back(cv::Point((int)(x+line_R[0]),(int)(y+line_R[1])));
            ptrs.push_back(points);


            cv::polylines(frame,points,true,cv::Scalar(100,220,60),8);
            cv::fillPoly(frame,ptrs,cv::Scalar(100,200,0));

            cv::addWeighted(imageUncharged,0.5,frame,0.5,1,frame);
        }
        /***********************************/

        for (unsigned int i = 0; i < detections.size(); ++i) {
            const vector<float>& d = detections[i];
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score>0.6){
//            cout << file << "_";
//            cout << std::setfill('0') << std::setw(6) << 0 << " ";
//            cout << static_cast<int>(d[1]) << " ";
//            cout << score << " ";
//            cout << static_cast<int>(d[3] * frame.cols) << " ";
//            cout << static_cast<int>(d[4] * frame.rows) << " ";
//            cout << static_cast<int>(d[5] * frame.cols) << " ";
//            cout << static_cast<int>(d[6] * frame.rows) << std::endl;
            CvPoint x1(d[3] * frame.cols,d[4] * frame.rows),x2(d[5] * frame.cols,d[6] * frame.rows);
            cv::rectangle(frame,x1,x2,cv::Scalar(240,30,0),4,1,0);

            stringstream stream;
            //stream << fixed << setprecision(4) << d[1];
            //string s = stream.str();
            stream << fixed << setprecision(2) << d[2];
            string tag = label[static_cast<int>(d[1])];
            tag = tag + " " + stream.str();
            cv::putText(frame,tag,CvPoint(d[3] * frame.cols,d[4] * frame.rows-10),2,1,cv::Scalar(240,30,0));

            }
        }

        cv::imshow("im",frame);
        cv::waitKey(1);

    }
    capture.release();
    return 0;
}

