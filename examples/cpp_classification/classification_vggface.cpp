/* ETRI SW-SoC Academy VGG Face Application Demo Template */
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <dirent.h>
#include <string.h>

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;


class Classifier {
    public:
        Classifier(const string& model_file,
                   const string& trained_file);

        std::vector<float> Predict(const cv::Mat& img);

    private:
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(const cv::Mat& img,
                        std::vector<cv::Mat>* input_channels);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
};


Classifier::Classifier(const string& model_file, const string& trained_file) {

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    //CHECK_EQ(net_->num_inputs(),1)<<"Network should have exactly one input.";
    //CHECK_EQ(net_->num_outputs(),1)<<"Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    //CHECK(num_channels_ == 3 || num_channels_ == 1)
    //    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    //SetMean(mean_file);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels); //convert img to caffe input

    net_->Forward();   //additional

    /* Copy the output layer to std::vector */
    shared_ptr< Blob<float> > output_layer = net_-> blob_by_name("fc8");
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
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
        resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::Mat avgimg(input_geometry_, CV_32FC3, cv::Scalar(93.5940,104.7624,129.1863)); 
  //  cv::Mat avgimg(sample_resized.rows,sample_resized.cols, CV_32FC3, cv::Scalar(93.5940,104.7624,129.1863)); 위에 꺼랑 밑에꺼랑 둘중 하나 써주면됨.
    cv::subtract(sample_float, avgimg, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network. ";

}

void Classifier::WrapInputLayer(vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height= input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i=0; i<input_layer->channels();++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}



int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << "deploy.prototxt network.caffemodel datafolder_path image_file" << std::endl;
        return 1;
    }

    string model_file   = argv[1];
    string trained_file = argv[2];
    Classifier classifier(model_file, trained_file);

    string folderpath   = argv[3];
    string targetname   = argv[4];

    std::cout << "---------- Prediction for "
              << targetname << " ----------" << std::endl;

    // load target image
    cv::Mat target_image = cv::imread(targetname, -1);
    cv::Mat target_float;
    target_image.convertTo(target_float, CV_32FC3);

   // cv::Mat target_normalized;
   // cv::Mat avgimg(target_image.rows, target_image.cols, CV_32FC3, cv::Scalar(93.5940,104.7624,129.1863));
   // cv::subtract(target_float, avgimg, target_normalized);

    std::vector<float> target_output = classifier.Predict(target_float);

    // Feature Extraction for target file


    // list files in the folder
    DIR *dir;
    dir = opendir(folderpath.c_str());
    string imgName;
    struct dirent *ent;

    if (dir != NULL) {
        while ( (ent = readdir(dir)) != NULL ) {
            imgName = ent->d_name;
            if (imgName.compare(".")!=0 && imgName.compare("..")!=0)
            {
                string aux;
                aux.append(folderpath);
                aux.append(imgName);
                std::cout << aux << std::endl;
                cv::Mat img = cv::imread(aux, -1);
                if (!img.data) {
                    std::cout << "Could not open or find the image" << std::endl;
                    return -1;
                }

                std::vector<float> img_output = classifier.Predict(img);

                // compute cosine similarity
                float in_prod = 0;
                for (int i=0; i<target_output.size(); i++)
                    in_prod += target_output[i]*img_output[i];
                double sim = in_prod/(norm(target_output, cv::NORM_L2)*norm(img_output, cv::NORM_L2));
                std::cout << imgName << " Similarity: "<< sim*100.0 << "%"<< std::endl;

            }
        }
    }

    return 0;
}
