#include "opencv2/ml.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "vector"
#include <iostream>
#include <cstdio>       /* printf */
#include <assert.h>     /* assert */
#include <cmath>
#include <algorithm>    /* random_shuffle */
#include <unordered_set>

using namespace std;
using namespace cv;

class Demosaic_ML {
    /* we'd better train 3 channels into 3 independent ML models, for the following reasons
     * 1. we might want to separate Y and chroma channels, to avoid chroma noises
     *    e.g.
     *      details in Y (by ML model)
     *      chroma just bicubic interp
     *    or
     *      like what we did in tonemap, details in intensity, keep color ratio unchanged (bicubic)
     * 2. SR framework can be transferred to DM easily
     *    just use different sampling pattern on different kNN model
     * 3. then it's very close to regression framework as well
     * 4. modularization: ML models can be decoupled from image filtering/processing/reconstruction
     *    ML models only see vectors/matrices
     */
    // TODO: 
    // DONE: clipping the range for kNN, reg (before/after adding details to the interp results)
    // channel cross-talk modeling for regression
    // fine tune kNN
    // calc train_error and test_error and overall error

public:
    enum ML_TYPE {
        ML_kNN,
        ML_LR,
        ML_SVR
    };


    /* ==================================================================
     *                       common parameters/methods
     * ==================================================================
     */
private:
    /*src data*/
    vector<string> file_list;
    vector<Mat> img_list;
    unordered_set<int> test_idx;

    ML_TYPE type;
    int out_grid_size;
    int patch_radius;
    int n_reg_train_samples;
    const int ds_ratio = 2;
    const int num_channels = 3;

public:
    Demosaic_ML(ML_TYPE type_, int grid_size_ = 12, int patch_radius_ = 6)
        : type(type_), out_grid_size(grid_size_), patch_radius(patch_radius_), n_reg_train_samples(0), knns()
    {
        if (out_grid_size%4 != 0) printf("Warning: grid_size is suggested to be a multiple of 4\n");
        if (patch_radius%2 != 0) printf("Warning: patch_radius is suggested to be a even number\n");
    }

    const vector<Mat>& get_whole_img_list() const {
        return img_list;
    }

    // TODO: test images also need to be cropped! (width/height are multiples of out_grid_size)

    void loadTrainImages(const unordered_set<int>& test_idx_) { // test_idx will be excluded
        test_idx = test_idx_;
        string fname = "..//test_dataset//kodim01.png";
        int n_kNN_train_samples = 0;
        n_reg_train_samples = 0;
        for (int i=1; i<=24; i++) {
            fname[23] = '0'+(i/10);
            fname[24] = '0'+(i%10);
            auto img = imread(fname);
            if (img.empty()) {
                printf("fail to load %s\n", fname.c_str());
                continue;
            }
            printf("load image: %s, size: %dx%dx%d\n", fname.c_str(), img.rows, img.cols, img.channels());
            assert(num_channels == img.channels());
            // crop images
            Mat float_img;
            img.convertTo(float_img, CV_32FC3);

            // Coordinates of the top-left corner. This is a default interpretation of Rect_.x and Rect_.y in OpenCV
            Mat cropped_img = float_img(Rect(0/* x */, 0/* y */, 
                        (img.cols/out_grid_size)*out_grid_size/*width*/, 
                        (img.rows/out_grid_size)*out_grid_size/*height*/));
            img_list.emplace_back(cropped_img*(1.0/255));
            file_list.emplace_back(fname);
            if (test_idx.find(i) != test_idx.end()) continue;
            n_kNN_train_samples += (img.cols/out_grid_size)*(img.rows/out_grid_size);
            n_reg_train_samples += (cropped_img.cols-2*patch_radius)*(cropped_img.rows-2*patch_radius);
        }
        printf("loaded %d images, %d kNN train samples, %d reg train samples\n", 
                img_list.size(), n_kNN_train_samples, n_reg_train_samples);
        
        knns = vector<KNN>(num_channels, KNN(n_kNN_train_samples));

    }

    vector<Mat> linearInterpImages(const vector<Mat>& test_bayer_ptns) const {
        vector<Mat> res_imgs;
        for (auto& bayer: test_bayer_ptns) {
            Mat res = bayer.clone();
            linearInterp(res);
            res_imgs.emplace_back(res);
        }
        return res_imgs;
    }

    void linearInterp(Mat& bayer) const {
        // assume bayer is a bayer pattern: R-G-G-B
        // and we are filling all missing channel info
        assert(bayer.channels() == num_channels);
        for (int r=1; r<bayer.rows-1; r++) {
            for (int c=1; c<bayer.cols-1; c++) {
                // fill in B
                if (r%2==1 && c%2==0) {        // at G(1), fill in R(2) and B(0)
                    bayer.at<Vec3f>(r, c).val[0] = bayer.at<Vec3f>(r, c-1).val[0]*0.5 + bayer.at<Vec3f>(r, c+1).val[0]*0.5;
                    bayer.at<Vec3f>(r, c).val[2] = bayer.at<Vec3f>(r-1, c).val[2]*0.5 + bayer.at<Vec3f>(r+1, c).val[2]*0.5;
                } else if (r%2==0 && c%2==1) { // at G(1), fill in R(2) and B(0)
                    bayer.at<Vec3f>(r, c).val[0] = bayer.at<Vec3f>(r-1, c).val[0]*0.5 + bayer.at<Vec3f>(r+1, c).val[0]*0.5;
                    bayer.at<Vec3f>(r, c).val[2] = bayer.at<Vec3f>(r, c-1).val[2]*0.5 + bayer.at<Vec3f>(r, c+1).val[2]*0.5;
                } else if (r%2==0 && c%2==0) { // at R(2), fill in G(1) and B(0)
                    bayer.at<Vec3f>(r, c).val[0] = 
                        bayer.at<Vec3f>(r-1, c-1).val[0]*0.25 + bayer.at<Vec3f>(r-1, c+1).val[0]*0.25 +
                        bayer.at<Vec3f>(r+1, c-1).val[0]*0.25 + bayer.at<Vec3f>(r+1, c+1).val[0]*0.25;
                    bayer.at<Vec3f>(r, c).val[1] = 
                        bayer.at<Vec3f>(r, c-1).val[1]*0.25 + bayer.at<Vec3f>(r, c+1).val[1]*0.25 +
                        bayer.at<Vec3f>(r-1, c).val[1]*0.25 + bayer.at<Vec3f>(r+1, c).val[1]*0.25;
                } else {                       // at B(0), fill in G(1) and R(2)
                    bayer.at<Vec3f>(r, c).val[1] = 
                        bayer.at<Vec3f>(r, c-1).val[1]*0.25 + bayer.at<Vec3f>(r, c+1).val[1]*0.25 +
                        bayer.at<Vec3f>(r-1, c).val[1]*0.25 + bayer.at<Vec3f>(r+1, c).val[1]*0.25;
                    bayer.at<Vec3f>(r, c).val[2] = 
                        bayer.at<Vec3f>(r-1, c-1).val[2]*0.25 + bayer.at<Vec3f>(r-1, c+1).val[2]*0.25 +
                        bayer.at<Vec3f>(r+1, c-1).val[2]*0.25 + bayer.at<Vec3f>(r+1, c+1).val[2]*0.25;
                }
                
            }
        }
        
        // fill in all 4 boundaries
        bayer.row(1).copyTo(bayer.row(0));
        bayer.row(bayer.rows-2).copyTo(bayer.row(bayer.rows-1));
        bayer.col(1).copyTo(bayer.col(0));
        bayer.col(bayer.cols-2).copyTo(bayer.col(bayer.cols-1));
        
    }

    void makeBayer(Mat& img, double sigma=0.01) {
        assert(img.channels() == num_channels);
        Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC3);
        randn(noise, 0, sigma);
        img += noise;
        auto is_b = [](int r, int c) { return r%2 && c%2; };
        auto is_g = [](int r, int c) { return (r%2 && !(c%2)) || (!(r%2) && c%2); };
        auto is_r = [](int r, int c) { return !(r%2) && !(c%2); };
        for (int r=0; r<img.rows; r++) {
            for (int c=0; c<img.cols; c++) {
                if (is_b(r,c)) { img.at<Vec3f>(r, c).val[1] = img.at<Vec3f>(r, c).val[2] = 0; }
                if (is_g(r,c)) { img.at<Vec3f>(r, c).val[0] = img.at<Vec3f>(r, c).val[2] = 0; }
                if (is_r(r,c)) { img.at<Vec3f>(r, c).val[0] = img.at<Vec3f>(r, c).val[1] = 0; } 
            }
        }
    }
    
private:
    void antiAliasing(Mat& src) {
        // in super-resolution problem, we are given small but not aliasing images, 
        // that's why we have to use antialiasing filter for data synthesis
        GaussianBlur(src, src, Size(5, 5), 1);
        // 2*LPF_BW <= Pix sampling rate --> PSF_duration >= 2*(2*Pix_Wid == 2 pixels == 1 period) (??)
        // so for a 5x5 Gaussian, we set sigma 1, to cover 95% in 1 dimension (approx 1 period)
    }
    
    void getDetailImg(Mat& src, Mat& dst, Mat& diff) {
        src.copyTo(dst);
        makeBayer(dst);  // NOTE: also add noise for training
        linearInterp(dst);
        // GaussianBlur(src, dst, Size(5, 5), 1); 
        diff = src - dst;
    }






    /* ==================================================================
     *                             kNN model
     * ==================================================================
     */
private:
    struct KNN {
        // http://docs.opencv.org/3.0-beta/modules/ml/doc/k_nearest_neighbors.html
        Mat train_samples;
        Mat responses_idx;
        vector<Mat> responses;
        Ptr<ml::KNearest> model;
        Ptr<ml::TrainData> tdata;

        int n_train_samples;

        KNN(int n_train):n_train_samples(n_train) {}

        void train(double train_ratio) {
            model = ml::KNearest::create();
            model->setDefaultK(16);
            model->setIsClassifier(true);

            // randomly select the samples for training
            vector<int> sampleIdx(train_samples.rows, 0);
            for (int i=0; i<(int)sampleIdx.size(); i++) sampleIdx[i] = i;
            random_shuffle(sampleIdx.begin(), sampleIdx.end());
            Mat sampIdx = Mat::zeros((int)train_samples.rows*train_ratio, 1, CV_32S);
            for (int i=0; i<sampIdx.rows; i++) sampIdx.at<int>(i, 0) = sampleIdx[i];
            
            printf("[KNN::train] total train samples(%d), train_ratio(%f)\n", train_samples.rows, train_ratio);
            // http://docs.opencv.org/3.0-beta/modules/ml/doc/mldata.html#traindata-create
            tdata = ml::TrainData::create(train_samples, ml::ROW_SAMPLE, responses_idx, noArray(), sampIdx);
            model->train(tdata);
        }
        void predict(const Mat& samples, int k, vector<Mat>& pred_patches) {
            pred_patches.clear();

            Mat neighborResponses = Mat::zeros(samples.rows, k, CV_32F);
            Mat results, dists;
            model->findNearest(samples, k, results, neighborResponses, dists);
            
            // weighted average the neighbor responses to form the final predictions
            for (int pidx=0; pidx<samples.rows; pidx++) {
                const double EXP_COEFF = 5;//10;
                double w = exp(-EXP_COEFF*dists.at<float>(pidx, 0));
                Mat pat_sum = responses[(int)neighborResponses.at<float>(pidx, 0)]*w;
                double w_sum = w;
                for (int i=1; i<k; i++) {
                    w = exp(-EXP_COEFF*dists.at<float>(pidx, i));
                    pat_sum = pat_sum + responses[(int)neighborResponses.at<float>(pidx, i)]*w;
                    w_sum += w;
                }
                // pat_sum = pat_sum/w_sum; // weighted sum, not average
                pat_sum = pat_sum/k; // just normalize by k
                pred_patches.emplace_back(pat_sum);
            }
        }
    };
    vector<KNN> knns; // for each channel

public:
    void createTrainSet_kNN() {
        int n_kNN_train_samples = knns.front().n_train_samples;
        // BGR format
        knns[0].train_samples = Mat::zeros(n_kNN_train_samples, out_grid_size*out_grid_size/(ds_ratio*ds_ratio), CV_32F);
        knns[1].train_samples = Mat::zeros(n_kNN_train_samples, out_grid_size*out_grid_size/(ds_ratio), CV_32F);
        knns[2].train_samples = Mat::zeros(n_kNN_train_samples, out_grid_size*out_grid_size/(ds_ratio*ds_ratio), CV_32F);
        Mat responses_idx = Mat::zeros(n_kNN_train_samples, 1, CV_32S); // map to sample_idx
        int sample_idx = 0;

        auto is_b = [&](int r, int c) { return r%2 && c%2; };
        auto is_g = [&](int r, int c) { return (r%2 && !(c%2)) || (!(r%2) && c%2); };
        auto is_r = [&](int r, int c) { return !(r%2) && !(c%2); };

        // use R-G-G-B bayer pattern
        for (int img_idx=0; img_idx<(int)img_list.size(); img_idx++) {
            if (test_idx.find(img_idx) != test_idx.end()) continue;

            auto& img = img_list[img_idx];
            Mat dst;
            Mat diff;
            // antiAliasing(img); // NOTE: it does change the original images!!!!
            getDetailImg(img, dst, diff);
            Mat mv[3];
            split(diff, mv);
            // imshow("src", img);
            imshow("train: interp result", dst);
            imshow("train: diff", diff);
            waitKey(30);

            for (int grid_r = 0; grid_r<img.rows/out_grid_size; grid_r++) {
                for (int grid_c = 0; grid_c<img.cols/out_grid_size; grid_c++) {
                    // each grid represents a training sample
                    int b_cnt = 0, g_cnt = 0, r_cnt = 0; // counters for current training sample
                    for (int r = out_grid_size*grid_r; r<out_grid_size*(grid_r+1); r++) {
                        for (int c = out_grid_size*grid_c; c<out_grid_size*(grid_c+1); c++) {
                            // collect training data for 3 channels
                            // B
                            if (is_b(r, c)) knns[0].train_samples.at<float>(sample_idx, b_cnt++) = img.at<Vec3f>(r, c).val[0];
                            // G
                            if (is_g(r, c)) knns[1].train_samples.at<float>(sample_idx, g_cnt++) = img.at<Vec3f>(r, c).val[1];
                            // R
                            if (is_r(r, c)) knns[2].train_samples.at<float>(sample_idx, r_cnt++) = img.at<Vec3f>(r, c).val[2];
                        }
                    }

                    assert(b_cnt == knns[0].train_samples.cols);
                    assert(g_cnt == knns[1].train_samples.cols);
                    assert(r_cnt == knns[2].train_samples.cols);

                    responses_idx.at<int>(sample_idx, 0) = sample_idx;
                    
                    // C++: Mat Mat::operator()(const Rect& roi) const, no matrix data is copied.
                    knns[0].responses.emplace_back(mv[0](Rect(out_grid_size*grid_c/*x*/, out_grid_size*grid_r/*y*/, 
                                    out_grid_size, out_grid_size)));
                    knns[1].responses.emplace_back(mv[1](Rect(out_grid_size*grid_c/*x*/, out_grid_size*grid_r/*y*/, 
                                    out_grid_size, out_grid_size)));
                    knns[2].responses.emplace_back(mv[2](Rect(out_grid_size*grid_c/*x*/, out_grid_size*grid_r/*y*/, 
                                    out_grid_size, out_grid_size)));

                    sample_idx++;
                } 
            }
        }
        assert(sample_idx == knns[0].train_samples.rows);
        knns[0].responses_idx = knns[1].responses_idx = knns[2].responses_idx = responses_idx;
        
        printf("%s DONE\n", __FUNCTION__);
    }
    
    void train_kNN(double train_ratio=0.7) {
        for (int ch=0; ch<num_channels; ch++) {
            knns[ch].train(train_ratio);
        }
    }
    
    vector<Mat> predictImages_kNN(const vector<Mat>& test_bayer_ptns, int k = 5) {
        printf("[%s] start\n", __FUNCTION__);
        vector<Mat> res_imgs;

        for (int i=0; i<(int)test_bayer_ptns.size(); i++) {
            res_imgs.emplace_back(predict_kNN(test_bayer_ptns[i], k, i));
        }
        return res_imgs;
    }

private:
    void addDetail(Mat& res, const Mat& pat_B, const Mat& pat_G, const Mat& pat_R, int grid_r, int grid_c){
        for (int r=0; r<out_grid_size; r++) {
            for (int c=0; c<out_grid_size; c++) {
                // B
                auto& v1 = res.at<Vec3f>(r+out_grid_size*grid_r, c+out_grid_size*grid_c).val[0];
                v1 += pat_B.at<float>(r, c);
                v1 = v1<0 ? 0 : v1>1 ? 1.0 : v1;

                // G
                auto& v2 = res.at<Vec3f>(r+out_grid_size*grid_r, c+out_grid_size*grid_c).val[1];
                v2 += pat_G.at<float>(r, c);
                v2 = v2<0 ? 0 : v2>1 ? 1.0 : v2;
                // R
                auto& v3 = res.at<Vec3f>(r+out_grid_size*grid_r, c+out_grid_size*grid_c).val[2];
                v3 += pat_R.at<float>(r, c);
                v3 = v3<0 ? 0 : v3>1 ? 1.0 : v3;
            }
        }
    }


    Mat predict_kNN(const Mat& bayer, int k, int img_id) {
        printf("[%s] start\n", __FUNCTION__);
        Mat res = bayer.clone(); // indep copy
        Mat det = Mat::zeros(res.rows, res.cols, CV_32FC3);
        printf("[%s] interpolating...\n", __FUNCTION__);
        linearInterp(res);

        int n_test_samples = (bayer.rows/out_grid_size)*(bayer.cols/out_grid_size);
        int test_sample_idx = 0;
        auto is_b = [&](int r, int c) { return r%2 && c%2; };
        auto is_g = [&](int r, int c) { return (r%2 && !(c%2)) || (!(r%2) && c%2); };
        auto is_r = [&](int r, int c) { return !(r%2) && !(c%2); };

        // BGR format
        Mat test_samples_B = Mat::zeros(n_test_samples, out_grid_size*out_grid_size/(ds_ratio*ds_ratio), CV_32F);
        Mat test_samples_G = Mat::zeros(n_test_samples, out_grid_size*out_grid_size/(ds_ratio), CV_32F);
        Mat test_samples_R = Mat::zeros(n_test_samples, out_grid_size*out_grid_size/(ds_ratio*ds_ratio), CV_32F);

        // fill in the details, for each patch
        
        // create test samples for kNN model
        printf("[%s] creating test samples\n", __FUNCTION__);
        for (int grid_r = 0; grid_r<bayer.rows/out_grid_size; grid_r++) {
            for (int grid_c = 0; grid_c<bayer.cols/out_grid_size; grid_c++) {
                // each grid represents a test sample
                int b_cnt = 0, g_cnt = 0, r_cnt = 0; // counters for current test sample
                for (int r = out_grid_size*grid_r; r<out_grid_size*(grid_r+1); r++) {
                    for (int c = out_grid_size*grid_c; c<out_grid_size*(grid_c+1); c++) {
                        // collect test input data for 3 channels
                        // B
                        if (is_b(r, c)) test_samples_B.at<float>(test_sample_idx, b_cnt++) = bayer.at<Vec3f>(r, c).val[0];
                        // G
                        if (is_g(r, c)) test_samples_G.at<float>(test_sample_idx, g_cnt++) = bayer.at<Vec3f>(r, c).val[1];
                        // R
                        if (is_r(r, c)) test_samples_R.at<float>(test_sample_idx, r_cnt++) = bayer.at<Vec3f>(r, c).val[2];
                    }
                }

                assert(b_cnt == test_samples_B.cols);
                assert(g_cnt == test_samples_G.cols);
                assert(r_cnt == test_samples_R.cols);
                
                test_sample_idx++;
            } 
        }
        assert(test_sample_idx == n_test_samples);

        // predict the details
        vector<Mat> pred_pat_B, pred_pat_G, pred_pat_R;
        printf("[%s] kNN predicting\n", __FUNCTION__);
        knns[0].predict(test_samples_B, k, pred_pat_B);
        knns[1].predict(test_samples_G, k, pred_pat_G);
        knns[2].predict(test_samples_R, k, pred_pat_R);
        
        auto clip = [](vector<Mat>& pats){
            const float rng = 0.1;
            for (auto& pat: pats) {
                for (int r=0; r<pat.rows; r++) {
                    for (int c=0; c<pat.cols; c++) {
                        auto& v = pat.at<float>(r, c);
                        v = v < -rng ? -rng : v > rng ? rng : v; 
                    }
                }
            }
        };
        clip(pred_pat_B);
        clip(pred_pat_G);
        clip(pred_pat_R);

        // combine details with interp results
        printf("[%s] combining interp with details\n", __FUNCTION__);
        const int N_GRID_COL = bayer.cols/out_grid_size;
        for (int i=0; i<n_test_samples; i++) {
            int grid_r = i/N_GRID_COL;
            int grid_c = i%N_GRID_COL;
            addDetail(res, pred_pat_B[i], pred_pat_G[i], pred_pat_R[i], grid_r, grid_c);
            addDetail(det, pred_pat_B[i], pred_pat_G[i], pred_pat_R[i], grid_r, grid_c);
        }
        char det_name[30] = {0};
        char res_name[30] = {0};
        sprintf(det_name, "knn_detail_map_i%d_k%d.png", img_id, k);
        sprintf(res_name, "knn_res_img_i%d_k%d.png", img_id, k);

        imwrite(det_name, det*255+128);
        imwrite(res_name, res*255);

        return res;
    }




    /* ==================================================================
     *                      linear regression model
     * ==================================================================
     */
private:
    struct LR {
        Mat train_samples;
        Mat responses;
        Ptr<ml::SVM> model;
        Ptr<ml::TrainData> tdata;
        Mat weights;

        int n_train_samples;

        LR(int n_train):n_train_samples(n_train) {}

        void train(double gamma, double epsilon, double C) {
            weights = (train_samples.t()*train_samples).inv(DECOMP_SVD)*train_samples.t()*responses;
#if 0
            // http://docs.opencv.org/3.0-beta/modules/ml/doc/support_vector_machines.html
            // ref settings: ~/opencv-3.1.0/samples/cpp/train_HOG.cpp
            model = ml::SVM::create();
            model->setKernel(ml::SVM::RBF);
            model->setGamma(gamma);
            model->setType(ml::SVM::EPS_SVR);
            model->setP(epsilon);
            model->setC(C);

            // http://docs.opencv.org/3.0-beta/modules/ml/doc/mldata.html#traindata-create
            tdata = ml::TrainData::create(train_samples, ml::ROW_SAMPLE, responses);
            model->train(tdata);
#endif
        }
        void predict(const Mat& samples, Mat& results) {
            for (int i=0; i<results.rows; i++) {
                results = samples*weights;
            }
#if 0
            model->predict(samples, results);
#endif
        }
    };
    vector<LR> lrs; // for each channel

    // define lrs model id
    #define B_Ver       (0)
    #define B_Hor       (1)
    #define B_4C        (2)

    #define G_All       (3)
    
    #define R_Ver       (4)
    #define R_Hor       (5)
    #define R_4C        (6)

    #define LRS_NUM     (7)


public:
    void createTrainSet_LR(double train_ratio = 0.1) {
        const int rand_base = 10000;
        // for B: vertical, horizontal, 4 corners
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound 
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound
        // for G
        lrs.emplace_back(n_reg_train_samples/2*train_ratio*4); // *4 to avoid out of bound
        // for R: same structures as B
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound
        lrs.emplace_back(n_reg_train_samples/4*train_ratio*4); // *4 to avoid out of bound


        // BGR format
        lrs[B_Ver].train_samples = Mat::zeros(lrs[B_Ver].n_train_samples, patch_radius*(patch_radius+1)+1, CV_32F);
        lrs[B_Hor].train_samples = Mat::zeros(lrs[B_Hor].n_train_samples, patch_radius*(patch_radius+1)+1, CV_32F);
        lrs[B_4C].train_samples = Mat::zeros(lrs[B_4C].n_train_samples, patch_radius*patch_radius+1, CV_32F);

        lrs[G_All].train_samples = Mat::zeros(lrs[G_All].n_train_samples, patch_radius*(2*patch_radius+2)+1, CV_32F);

        lrs[R_Ver].train_samples = Mat::zeros(lrs[R_Ver].n_train_samples, patch_radius*(patch_radius+1)+1, CV_32F);
        lrs[R_Hor].train_samples = Mat::zeros(lrs[R_Hor].n_train_samples, patch_radius*(patch_radius+1)+1, CV_32F);
        lrs[R_4C].train_samples = Mat::zeros(lrs[R_4C].n_train_samples, patch_radius*patch_radius+1, CV_32F);


        lrs[B_Ver].responses = Mat::zeros(lrs[B_Ver].n_train_samples, 1, CV_32F);
        lrs[B_Hor].responses = Mat::zeros(lrs[B_Hor].n_train_samples, 1, CV_32F);
        lrs[B_4C].responses = Mat::zeros(lrs[B_4C].n_train_samples, 1, CV_32F);

        lrs[G_All].responses = Mat::zeros(lrs[G_All].n_train_samples, 1, CV_32F);

        lrs[R_Ver].responses = Mat::zeros(lrs[R_Ver].n_train_samples, 1, CV_32F);
        lrs[R_Hor].responses = Mat::zeros(lrs[R_Hor].n_train_samples, 1, CV_32F);
        lrs[R_4C].responses = Mat::zeros(lrs[R_4C].n_train_samples, 1, CV_32F);

        printf("[%s] lrs setup done\n", __FUNCTION__);
        auto is_b = [&](int r, int c) { return r%2 && c%2; };
        auto is_g = [&](int r, int c) { return (r%2 && !(c%2)) || (!(r%2) && c%2); };
        auto is_r = [&](int r, int c) { return !(r%2) && !(c%2); };

        // use R-G-G-B bayer pattern
        int sample_idx_G=0, sample_idx_BV=0, sample_idx_BH=0, sample_idx_B4=0, sample_idx_RV=0, sample_idx_RH=0, sample_idx_R4=0;
        for (int img_idx=0; img_idx<(int)img_list.size(); img_idx++) {
            if (test_idx.find(img_idx) != test_idx.end()) continue;
            auto& img = img_list[img_idx];
            Mat dst;
            Mat diff;
            // antiAliasing(img); // NOTE: it does change the original images!!!!
            getDetailImg(img, dst, diff);
            Mat mv[3];
            split(diff, mv);
            // imshow("src", img);
            imshow("train: interp result", dst);
            imshow("train: diff", diff);
            printf("[%s] train_ratio(%f), processing img_idx(%d), BV(%d), BH(%d), B4(%d), G(%d), RV(%d), RH(%d), R4(%d)\n"
                    , __FUNCTION__, train_ratio
                    , img_idx, sample_idx_BV, sample_idx_BH, sample_idx_B4, sample_idx_G, sample_idx_RV
                    , sample_idx_RH, sample_idx_R4);
            waitKey(30);

            for (int ctr_r = 0+patch_radius; ctr_r<img.rows-patch_radius; ctr_r++) {
                for (int ctr_c = 0+patch_radius; ctr_c<img.cols-patch_radius; ctr_c++) {
                    if (rand()%rand_base >= rand_base*train_ratio) continue;

                    int b_cnt = 0, g_cnt = 0, r_cnt = 0; // counters for current training sample
                    if (ctr_r%2==0 && ctr_c%2==0) { // at R, B_4C
                        for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                            for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                                // B
                                if (is_b(r, c)) lrs[B_4C].train_samples.at<float>(sample_idx_B4, b_cnt++) = img.at<Vec3f>(r, c).val[0];
                                // G
                                if (is_g(r, c)) lrs[G_All].train_samples.at<float>(sample_idx_G, g_cnt++) = img.at<Vec3f>(r, c).val[1];
                            }
                        }
                        lrs[B_4C].train_samples.at<float>(sample_idx_B4, b_cnt++) = 1;
                        lrs[G_All].train_samples.at<float>(sample_idx_G, g_cnt++) = 1;
                        assert(b_cnt == lrs[B_4C].train_samples.cols);
                        assert(g_cnt == lrs[G_All].train_samples.cols);
                        lrs[B_4C].responses.at<float>(sample_idx_B4, 0) = mv[0].at<float>(ctr_r, ctr_c);
                        lrs[G_All].responses.at<float>(sample_idx_G, 0) = mv[1].at<float>(ctr_r, ctr_c);
                        sample_idx_B4++;
                        sample_idx_G++;
                    } else if (ctr_r%2==1 && ctr_c%2==1) { // at B, R_4C
                        for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                            for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                                // R
                                if (is_r(r, c)) lrs[R_4C].train_samples.at<float>(sample_idx_R4, r_cnt++) = img.at<Vec3f>(r, c).val[2];
                                // G
                                if (is_g(r, c)) lrs[G_All].train_samples.at<float>(sample_idx_G, g_cnt++) = img.at<Vec3f>(r, c).val[1];
                            }
                        }
                        lrs[R_4C].train_samples.at<float>(sample_idx_R4, r_cnt++) = 1;
                        lrs[G_All].train_samples.at<float>(sample_idx_G, g_cnt++) = 1;
                        assert(r_cnt == lrs[R_4C].train_samples.cols);
                        assert(g_cnt == lrs[G_All].train_samples.cols);
                        lrs[R_4C].responses.at<float>(sample_idx_R4, 0) = mv[2].at<float>(ctr_r, ctr_c);
                        lrs[G_All].responses.at<float>(sample_idx_G, 0) = mv[1].at<float>(ctr_r, ctr_c);
                        sample_idx_R4++;
                        sample_idx_G++;
                    } else if (ctr_r%2==1 && ctr_c%2==0) { // at G, R_Ver, B_Hor
                        for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                            for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                                // B
                                if (is_b(r, c)) lrs[B_Hor].train_samples.at<float>(sample_idx_BH, b_cnt++) = img.at<Vec3f>(r, c).val[0];
                                // R
                                if (is_r(r, c)) lrs[R_Ver].train_samples.at<float>(sample_idx_RV, r_cnt++) = img.at<Vec3f>(r, c).val[2];
                            }
                        }
                        lrs[B_Hor].train_samples.at<float>(sample_idx_BH, b_cnt++) = 1;
                        lrs[R_Ver].train_samples.at<float>(sample_idx_RV, r_cnt++) = 1;
                        assert(b_cnt == lrs[B_Hor].train_samples.cols);
                        assert(r_cnt == lrs[R_Ver].train_samples.cols);
                        lrs[B_Hor].responses.at<float>(sample_idx_BH, 0) = mv[0].at<float>(ctr_r, ctr_c);
                        lrs[R_Ver].responses.at<float>(sample_idx_RV, 0) = mv[2].at<float>(ctr_r, ctr_c);
                        sample_idx_BH++;
                        sample_idx_RV++;
                    } else { // at G, R_Hor, B_Ver
                        for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                            for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                                // R
                                if (is_r(r, c)) lrs[R_Hor].train_samples.at<float>(sample_idx_RH, r_cnt++) = img.at<Vec3f>(r, c).val[2];
                                // B
                                if (is_b(r, c)) lrs[B_Ver].train_samples.at<float>(sample_idx_BV, b_cnt++) = img.at<Vec3f>(r, c).val[0];
                            }
                        }
                        lrs[R_Hor].train_samples.at<float>(sample_idx_RH, r_cnt++) = 1;
                        lrs[B_Ver].train_samples.at<float>(sample_idx_BV, b_cnt++) = 1;
                        assert(r_cnt == lrs[R_Hor].train_samples.cols);
                        assert(b_cnt == lrs[B_Ver].train_samples.cols);
                        lrs[R_Hor].responses.at<float>(sample_idx_RH, 0) = mv[2].at<float>(ctr_r, ctr_c);
                        lrs[B_Ver].responses.at<float>(sample_idx_BV, 0) = mv[0].at<float>(ctr_r, ctr_c);
                        sample_idx_RH++;
                        sample_idx_BV++;
                    }
                    
                } 
            }
        }
        // now crop the training set with the actual number of train samples
        lrs[B_Ver].train_samples = lrs[B_Ver].train_samples(Rect(0, 0, lrs[B_Ver].train_samples.cols, sample_idx_BV));
        lrs[B_Hor].train_samples = lrs[B_Hor].train_samples(Rect(0, 0, lrs[B_Hor].train_samples.cols, sample_idx_BH));
        lrs[B_4C].train_samples = lrs[B_4C].train_samples(Rect(0, 0, lrs[B_4C].train_samples.cols, sample_idx_B4));

        lrs[G_All].train_samples = lrs[G_All].train_samples(Rect(0, 0, lrs[G_All].train_samples.cols, sample_idx_G));

        lrs[R_Ver].train_samples = lrs[R_Ver].train_samples(Rect(0, 0, lrs[R_Ver].train_samples.cols, sample_idx_RV));
        lrs[R_Hor].train_samples = lrs[R_Hor].train_samples(Rect(0, 0, lrs[R_Hor].train_samples.cols, sample_idx_RH));
        lrs[R_4C].train_samples = lrs[R_4C].train_samples(Rect(0, 0, lrs[R_4C].train_samples.cols, sample_idx_R4));
        
        lrs[B_Ver].responses = lrs[B_Ver].responses(Rect(0, 0, lrs[B_Ver].responses.cols, sample_idx_BV));
        lrs[B_Hor].responses = lrs[B_Hor].responses(Rect(0, 0, lrs[B_Hor].responses.cols, sample_idx_BH));
        lrs[B_4C].responses = lrs[B_4C].responses(Rect(0, 0, lrs[B_4C].responses.cols, sample_idx_B4));
        
        lrs[G_All].responses = lrs[G_All].responses(Rect(0, 0, lrs[G_All].responses.cols, sample_idx_G));

        lrs[R_Ver].responses = lrs[R_Ver].responses(Rect(0, 0, lrs[R_Ver].responses.cols, sample_idx_RV));
        lrs[R_Hor].responses = lrs[R_Hor].responses(Rect(0, 0, lrs[R_Hor].responses.cols, sample_idx_RH));
        lrs[R_4C].responses = lrs[R_4C].responses(Rect(0, 0, lrs[R_4C].responses.cols, sample_idx_R4));
        
        printf("%s DONE\n", __FUNCTION__);
    }
   
    void train_LR(double gamma=0.1, double epsilon=0.1, double C=20) {
        for (int ch=0; ch<LRS_NUM; ch++) {
            printf("[%s] training lrs[%d]...\n", __FUNCTION__, ch);
            lrs[ch].train(gamma, epsilon, C);
        }
    }
    
    vector<Mat> predictImages_LR(const vector<Mat>& test_bayer_ptns) {
        printf("[%s] start\n", __FUNCTION__);
        vector<Mat> res_imgs;

        for (int i=0; i<(int)test_bayer_ptns.size(); i++) {
            res_imgs.emplace_back(predict_LR(test_bayer_ptns[i], i));
        }
        return res_imgs;
    }

private:

    Mat predict_LR(const Mat& bayer, int img_id) {
        printf("[%s] start\n", __FUNCTION__);
        Mat res = bayer.clone(); // indep copy
        Mat det = Mat::zeros(res.rows, res.cols, CV_32FC3);
        printf("[%s] interpolating...\n", __FUNCTION__);
        linearInterp(res);

        // BGR format
        Mat test_samples_BV = Mat::zeros(1, lrs[B_Ver].train_samples.cols, CV_32F);
        Mat test_samples_BH = Mat::zeros(1, lrs[B_Hor].train_samples.cols, CV_32F);
        Mat test_samples_B4 = Mat::zeros(1, lrs[B_4C].train_samples.cols, CV_32F);

        Mat test_samples_G = Mat::zeros(1, lrs[G_All].train_samples.cols, CV_32F);

        Mat test_samples_RV = Mat::zeros(1, lrs[R_Ver].train_samples.cols, CV_32F);
        Mat test_samples_RH = Mat::zeros(1, lrs[R_Hor].train_samples.cols, CV_32F);
        Mat test_samples_R4 = Mat::zeros(1, lrs[R_4C].train_samples.cols, CV_32F);

        auto is_b = [](int r, int c) { return r%2 && c%2; };
        auto is_g = [](int r, int c) { return (r%2 && !(c%2)) || (!(r%2) && c%2); };
        auto is_r = [](int r, int c) { return !(r%2) && !(c%2); };
        
        const float rng = 0.2;
        auto clip = [](float v, float rng) { return v < -rng ? -rng : v > rng ? rng : v; };
        for (int ctr_r = 0+patch_radius; ctr_r<bayer.rows-patch_radius; ctr_r++) {
            for (int ctr_c = 0+patch_radius; ctr_c<bayer.cols-patch_radius; ctr_c++) {
                // printf("ctr_r(%d), ctr_c(%d)\n", ctr_r, ctr_c);
                int b_cnt = 0, g_cnt = 0, r_cnt = 0; // counters for current training sample
                if (ctr_r%2==0 && ctr_c%2==0) { // at R, B_4C
                    for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                        for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                            // B
                            if (is_b(r, c)) test_samples_B4.at<float>(0, b_cnt++) = bayer.at<Vec3f>(r, c).val[0];
                            // G
                            if (is_g(r, c)) test_samples_G.at<float>(0, g_cnt++) = bayer.at<Vec3f>(r, c).val[1];
                        }
                    }
                    test_samples_B4.at<float>(0, b_cnt++) = 1;
                    test_samples_G.at<float>(0, g_cnt++) = 1;
                    assert(b_cnt == test_samples_B4.cols);
                    assert(g_cnt == test_samples_G.cols);
                    Mat v = Mat::zeros(1, 1, CV_32F);
                    lrs[B_4C].predict(test_samples_B4, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[0] = clip(v.at<float>(0, 0), rng);
                    lrs[G_All].predict(test_samples_G, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[1] = clip(v.at<float>(0, 0), rng);
                } else if (ctr_r%2==1 && ctr_c%2==1) { // at B, R_4C
                    for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                        for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                            // R
                            if (is_r(r, c)) test_samples_R4.at<float>(0, r_cnt++) = bayer.at<Vec3f>(r, c).val[2];
                            // G
                            if (is_g(r, c)) test_samples_G.at<float>(0, g_cnt++) = bayer.at<Vec3f>(r, c).val[1];
                        }
                    }
                    test_samples_R4.at<float>(0, r_cnt++) = 1;
                    test_samples_G.at<float>(0, g_cnt++) = 1;
                    assert(r_cnt == test_samples_R4.cols);
                    assert(g_cnt == test_samples_G.cols);
                    Mat v = Mat::zeros(1, 1, CV_32F);
                    lrs[R_4C].predict(test_samples_R4, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[2] = clip(v.at<float>(0, 0), rng);
                    lrs[G_All].predict(test_samples_G, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[1] = clip(v.at<float>(0, 0), rng);
                } else if (ctr_r%2==1 && ctr_c%2==0) { // at G, R_Ver, B_Hor
                    for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                        for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                            // B
                            if (is_b(r, c)) test_samples_BH.at<float>(0, b_cnt++) = bayer.at<Vec3f>(r, c).val[0];
                            // R
                            if (is_r(r, c)) test_samples_RV.at<float>(0, r_cnt++) = bayer.at<Vec3f>(r, c).val[2];
                        }
                    }
                    test_samples_BH.at<float>(0, b_cnt++) = 1;
                    test_samples_RV.at<float>(0, r_cnt++) = 1;
                    assert(b_cnt == test_samples_BH.cols);
                    assert(r_cnt == test_samples_RV.cols);
                    Mat v = Mat::zeros(1, 1, CV_32F);
                    lrs[B_Hor].predict(test_samples_BH, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[0] = clip(v.at<float>(0, 0), rng);
                    lrs[R_Ver].predict(test_samples_RV, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[2] = clip(v.at<float>(0, 0), rng);
                } else { // at G, R_Hor, B_Ver
                    for (int r=ctr_r-patch_radius; r<=ctr_r+patch_radius; r++) {
                        for (int c=ctr_c-patch_radius; c<=ctr_c+patch_radius; c++) {
                            // R
                            if (is_r(r, c)) test_samples_RH.at<float>(0, r_cnt++) = bayer.at<Vec3f>(r, c).val[2];
                            // B
                            if (is_b(r, c)) test_samples_BV.at<float>(0, b_cnt++) = bayer.at<Vec3f>(r, c).val[0];
                        }
                    }
                    test_samples_RH.at<float>(0, r_cnt++) = 1;
                    test_samples_BV.at<float>(0, b_cnt++) = 1;
                    assert(r_cnt == test_samples_RH.cols);
                    assert(b_cnt == test_samples_BV.cols);
                    Mat v = Mat::zeros(1, 1, CV_32F);
                    lrs[R_Hor].predict(test_samples_RH, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[2] = clip(v.at<float>(0, 0), rng);
                    lrs[B_Ver].predict(test_samples_BV, v);
                    det.at<Vec3f>(ctr_r, ctr_c).val[0] = clip(v.at<float>(0, 0), rng);
                }   
            } 
        }

        // combine details with interp results
        printf("[%s] combining interp with details\n", __FUNCTION__);
        res = res + det;
        for (int r=0; r<res.rows; r++) {
            for (int c=0; c<res.cols; c++) {
                res.at<Vec3f>(r, c).val[0] = clip(res.at<Vec3f>(r, c).val[0], 1.0);
                res.at<Vec3f>(r, c).val[1] = clip(res.at<Vec3f>(r, c).val[1], 1.0);
                res.at<Vec3f>(r, c).val[2] = clip(res.at<Vec3f>(r, c).val[2], 1.0);
            }
        }

        char det_name[30] = {0};
        char res_name[30] = {0};
        sprintf(det_name, "lr_detail_map_i%d_p%d.png", img_id, patch_radius);
        sprintf(res_name, "lr_res_img_i%d_p%d.png", img_id, patch_radius);

        imwrite(det_name, det*255+128);
        imwrite(res_name, res*255);

        return res;
    }



};


void PSNR(const Mat& pred, const Mat& ref, double& psnr_db, double& mse, double& peak, int crop_margin=0) {
    assert(pred.cols==ref.cols && pred.rows==ref.rows && pred.channels()==ref.channels());
    assert(ref.channels() == 3);
    double src_peak = 0;
    mse = 0;
    peak = 0;
    for (int r=0+crop_margin; r<pred.rows-crop_margin; r++) {
        for (int c=0+crop_margin; c<pred.cols-crop_margin; c++) {
            for (int ch=0; ch<3; ch++) {
                mse += pow(pred.at<Vec3f>(r, c).val[ch] - ref.at<Vec3f>(r, c).val[ch], 2.0);
                if (peak < pred.at<Vec3f>(r, c).val[ch]) peak = pred.at<Vec3f>(r, c).val[ch];
                if (src_peak < ref.at<Vec3f>(r, c).val[ch]) src_peak = ref.at<Vec3f>(r, c).val[ch];
            }
        }
    }
    mse /= ((pred.rows-2*crop_margin)*(pred.cols-2*crop_margin)*3);
    psnr_db = 10*log10(peak*peak/mse);
    printf("src_peak: %f\n", src_peak);
}

int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;
    unordered_set<int> test_idx{4, 7, 20, 22}; //0-based, will be excluded from the training process
    for (int ps=8/*8*/; ps<=8/*24*/; ps+=4) {
        Demosaic_ML dm_engine(Demosaic_ML::ML_TYPE::ML_kNN, ps/*12*/, ps/2/*6*/);
        dm_engine.loadTrainImages(test_idx);
        // image = Mat::zeros(480, 640, CV_8UC1);
        /*
        for(int i=0; i<(int)dm_engine.get_train_img_list().size(); i++)
        {
            cout << i << endl;
            imshow("Sample", dm_engine.get_train_img_list()[i]);
            if(waitKey(100) >= 0)
                break;
        }
        */
        // create the test set
        vector<Mat> test_img_set;
        vector<Mat> test_ground_truth;
#if 0 // evaluate only on the test set
        for (auto i: test_idx) {
            Mat bayer = dm_engine.get_whole_img_list()[i].clone();
            dm_engine.makeBayer(bayer);
            test_img_set.emplace_back(bayer);
            test_ground_truth.emplace_back(dm_engine.get_whole_img_list()[i]); // not cloning
        }
#else // evaluate on the whole image set
        for (int i=0; i<(int)dm_engine.get_whole_img_list().size(); i++) {
            Mat bayer = dm_engine.get_whole_img_list()[i].clone();
            dm_engine.makeBayer(bayer);
            test_img_set.emplace_back(bayer);
            test_ground_truth.emplace_back(dm_engine.get_whole_img_list()[i]); // not cloning
        }
#endif
        printf("test_img_set created\n");
        
        
        double psnr, mse, pk;
        auto res_imgs = dm_engine.linearInterpImages(test_img_set);
        double sum_mse=0;
        for(int i=0; i<(int)res_imgs.size(); i++)
        {
            PSNR(res_imgs[i], test_ground_truth[i], psnr, mse, pk, 1);
            printf("bilinear psnr(dB): %f, mse: %f, pk: %f\n", psnr, mse, pk);
            sum_mse += mse;
            /*
            imshow("Sample", res_imgs[i]);
            if(waitKey(500) >= 0)
                break;
            */
        }
        printf("interp sum_mse: %f\n", sum_mse);

        dm_engine.createTrainSet_kNN();
        dm_engine.train_kNN();
        for (int k=18/*3*/; k<=18; k+=3) {
            sum_mse = 0;
            res_imgs = dm_engine.predictImages_kNN(test_img_set, k);
            for(int i=0; i<(int)res_imgs.size(); i++)
            {
                PSNR(res_imgs[i], test_ground_truth[i], psnr, mse, pk, 1);
                printf("kNN(k=%d, ps=%d) psnr(dB): %f, mse: %f, pk: %f\n", k, ps, psnr, mse, pk);
                sum_mse += mse;
                /*
                imshow("Sample", res_imgs[i]);
                if(waitKey(500) >= 0)
                    break;
                */
            }
            printf("kNN(k=%d, ps=%d) sum_mse: %f\n", k, ps, sum_mse);
        }

        dm_engine.createTrainSet_LR();
        dm_engine.train_LR();
        res_imgs = dm_engine.predictImages_LR(test_img_set);
        sum_mse = 0;
        for(int i=0; i<(int)res_imgs.size(); i++)
        {
            PSNR(res_imgs[i], test_ground_truth[i], psnr, mse, pk, 6);
            printf("reg(ps=%d) psnr(dB): %f, mse: %f, pk: %f\n", ps, psnr, mse, pk);
            sum_mse += mse;
            /*
            imshow("Sample", res_imgs[i]);
            if(waitKey(500) >= 0)
                break;
            */
        }
        printf("reg(ps=%d) sum_mse: %f\n", ps, sum_mse);
    }
    return 0;
}

