This is the README file for opencv code (kNN & linear regression method)

To install the OpenCV package, please follow the instructions in the following webpage
http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html
http://opencv.org/downloads.html
(Sorry, I didn't find time to test if my executable and opencv's .so files can be run on 
other OS system (mine is ubuntu linux 16.04 LTS). Besides, OpenCV 3.0 is not well documented, 
can't find any document mentioning people can run out of box on anther OS for OpenCV 3.)

Assume you have installed OpenCV 3 and compiled my code 
(or the executable can be run out of box)

How to compile:
$ make

How to run it:
$ ./opencv_knn_svr_dm

the sample output is as below, feel free to reach out to me if you have any question
Now the main function is using the optimal settings, 
but you can play around with the parameters such as 
in opencv_knn_svr_dm.cpp line 908, you can change k's range (within 3-18) 
in opencv_knn_svr_dm.cpp line 856, you can change ps's range (within 8-20, must be 4's multiple)

Thanks!!



----------------------------------------------------------------------------------------







idfs@idfs-K42Jr:~/GitHubWorkSpace/367-Project/opencv_cpp$ ./opencv_knn_svr_dm 
Built with OpenCV 3.1.0
load image: ..//test_dataset//kodim01.png, size: 512x768x3
load image: ..//test_dataset//kodim02.png, size: 512x768x3
load image: ..//test_dataset//kodim03.png, size: 512x768x3
load image: ..//test_dataset//kodim04.png, size: 768x512x3
load image: ..//test_dataset//kodim05.png, size: 512x768x3
load image: ..//test_dataset//kodim06.png, size: 512x768x3
load image: ..//test_dataset//kodim07.png, size: 512x768x3
load image: ..//test_dataset//kodim08.png, size: 512x768x3
load image: ..//test_dataset//kodim09.png, size: 768x512x3
load image: ..//test_dataset//kodim10.png, size: 768x512x3
load image: ..//test_dataset//kodim11.png, size: 512x768x3
load image: ..//test_dataset//kodim12.png, size: 512x768x3
load image: ..//test_dataset//kodim13.png, size: 512x768x3
load image: ..//test_dataset//kodim14.png, size: 512x768x3
load image: ..//test_dataset//kodim15.png, size: 512x768x3
load image: ..//test_dataset//kodim16.png, size: 512x768x3
load image: ..//test_dataset//kodim17.png, size: 768x512x3
load image: ..//test_dataset//kodim18.png, size: 768x512x3
load image: ..//test_dataset//kodim19.png, size: 768x512x3
load image: ..//test_dataset//kodim20.png, size: 512x768x3
load image: ..//test_dataset//kodim21.png, size: 512x768x3
load image: ..//test_dataset//kodim22.png, size: 512x768x3
load image: ..//test_dataset//kodim23.png, size: 512x768x3
load image: ..//test_dataset//kodim24.png, size: 512x768x3
loaded 24 images, 122880 kNN train samples, 7660800 reg train samples
test_img_set created
src_peak: 1.000000
bilinear psnr(dB): 26.329916, mse: 0.002469, pk: 1.029711
src_peak: 1.000000
bilinear psnr(dB): 32.610824, mse: 0.000576, pk: 1.025121
src_peak: 1.000000
bilinear psnr(dB): 33.704168, mse: 0.000466, pk: 1.046093
src_peak: 1.000000
bilinear psnr(dB): 33.223086, mse: 0.000514, pk: 1.039052
src_peak: 1.000000
bilinear psnr(dB): 26.804706, mse: 0.002223, pk: 1.031950
src_peak: 1.000000
bilinear psnr(dB): 27.848566, mse: 0.001780, pk: 1.041383
src_peak: 1.000000
bilinear psnr(dB): 32.851091, mse: 0.000538, pk: 1.018766
src_peak: 1.000000
bilinear psnr(dB): 23.938392, mse: 0.004350, pk: 1.037914
src_peak: 1.000000
bilinear psnr(dB): 32.210178, mse: 0.000634, pk: 1.027224
src_peak: 1.000000
bilinear psnr(dB): 32.146447, mse: 0.000650, pk: 1.031974
src_peak: 1.000000
bilinear psnr(dB): 29.305951, mse: 0.001242, pk: 1.028916
src_peak: 1.000000
bilinear psnr(dB): 32.731291, mse: 0.000578, pk: 1.041341
src_peak: 1.000000
bilinear psnr(dB): 24.170002, mse: 0.004103, pk: 1.035213
src_peak: 1.000000
bilinear psnr(dB): 29.247682, mse: 0.001276, pk: 1.035720
src_peak: 1.000000
bilinear psnr(dB): 31.223358, mse: 0.000813, pk: 1.038073
src_peak: 1.000000
bilinear psnr(dB): 30.986686, mse: 0.000851, pk: 1.033176
src_peak: 1.000000
bilinear psnr(dB): 31.868715, mse: 0.000696, pk: 1.034582
src_peak: 1.000000
bilinear psnr(dB): 28.106316, mse: 0.001654, pk: 1.034159
src_peak: 1.000000
bilinear psnr(dB): 28.294302, mse: 0.001587, pk: 1.034992
src_peak: 1.000000
bilinear psnr(dB): 31.377297, mse: 0.000792, pk: 1.042856
src_peak: 1.000000
bilinear psnr(dB): 28.680329, mse: 0.001450, pk: 1.034361
src_peak: 1.000000
bilinear psnr(dB): 30.655874, mse: 0.000933, pk: 1.041528
src_peak: 1.000000
bilinear psnr(dB): 34.552965, mse: 0.000374, pk: 1.033524
src_peak: 1.000000
bilinear psnr(dB): 26.981515, mse: 0.002180, pk: 1.043120
interp sum_mse: 0.032727
createTrainSet_kNN DONE
[KNN::train] total train samples(122880), train_ratio(0.700000)
[KNN::train] total train samples(122880), train_ratio(0.700000)
[KNN::train] total train samples(122880), train_ratio(0.700000)
[predictImages_kNN] start
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
[predict_kNN] start
[predict_kNN] interpolating...
[predict_kNN] creating test samples
[predict_kNN] kNN predicting
[predict_kNN] combining interp with details
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 26.618176, mse: 0.002179, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 32.982610, mse: 0.000503, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 34.104121, mse: 0.000389, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 33.391377, mse: 0.000458, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 26.663508, mse: 0.002156, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 28.086302, mse: 0.001554, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 33.433922, mse: 0.000454, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 23.664492, mse: 0.004301, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 32.508442, mse: 0.000561, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 32.381248, mse: 0.000578, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 29.475488, mse: 0.001128, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 33.274806, mse: 0.000470, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 24.242765, mse: 0.003765, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 29.458826, mse: 0.001133, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 32.139985, mse: 0.000611, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 31.268345, mse: 0.000747, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 32.142405, mse: 0.000611, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 28.139462, mse: 0.001535, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 28.645907, mse: 0.001366, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 31.660527, mse: 0.000682, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 28.405545, mse: 0.001444, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 30.639252, mse: 0.000863, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 34.297591, mse: 0.000372, pk: 1.000000
src_peak: 1.000000
kNN(k=18, ps=8) psnr(dB): 26.954394, mse: 0.002016, pk: 1.000000
kNN(k=18, ps=8) sum_mse: 0.029874
[createTrainSet_LR] lrs setup done
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(0), BV(0), BH(0), B4(0), G(0), RV(0), RH(0), R4(0)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(1), BV(9760), BH(9643), B4(9642), G(19169), RV(9643), RH(9760), R4(9527)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(2), BV(19304), BH(19048), B4(19167), G(38233), RV(19048), RH(19304), R4(19066)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(3), BV(28640), BH(28648), B4(28768), G(57496), RV(28648), RH(28640), R4(28728)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(5), BV(38068), BH(38247), B4(38314), G(76494), RV(38247), RH(38068), R4(38180)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(6), BV(47542), BH(47837), B4(47942), G(95796), RV(47837), RH(47542), R4(47854)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(8), BV(57079), BH(57610), B4(57594), G(115137), RV(57610), RH(57079), R4(57543)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(9), BV(66728), BH(67347), B4(67232), G(134428), RV(67347), RH(66728), R4(67196)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(10), BV(76354), BH(76809), B4(76861), G(153697), RV(76809), RH(76354), R4(76836)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(11), BV(85773), BH(86549), B4(86743), G(173106), RV(86549), RH(85773), R4(86363)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(12), BV(95402), BH(96110), B4(96211), G(192196), RV(96110), RH(95402), R4(95985)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(13), BV(105108), BH(105746), B4(105903), G(211635), RV(105746), RH(105108), R4(105732)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(14), BV(114715), BH(115303), B4(115595), G(230771), RV(115303), RH(114715), R4(115176)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(15), BV(124382), BH(124831), B4(125327), G(250068), RV(124831), RH(124382), R4(124741)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(16), BV(133954), BH(134294), B4(134994), G(269473), RV(134294), RH(133954), R4(134479)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(17), BV(143472), BH(143843), B4(144550), G(288485), RV(143843), RH(143472), R4(143935)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(18), BV(153068), BH(153346), B4(154118), G(307673), RV(153346), RH(153068), R4(153555)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(19), BV(162645), BH(163033), B4(163549), G(326711), RV(163033), RH(162645), R4(163162)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(21), BV(172444), BH(172705), B4(173156), G(345990), RV(172705), RH(172444), R4(172834)
[createTrainSet_LR] train_ratio(0.100000), processing img_idx(23), BV(181977), BH(182410), B4(182760), G(365155), RV(182410), RH(181977), R4(182395)
createTrainSet_LR DONE
[train_LR] training lrs[0]...
[train_LR] training lrs[1]...
[train_LR] training lrs[2]...
[train_LR] training lrs[3]...
[train_LR] training lrs[4]...
[train_LR] training lrs[5]...
[train_LR] training lrs[6]...
[predictImages_LR] start
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
[predict_LR] start
[predict_LR] interpolating...
[predict_LR] combining interp with details
src_peak: 1.000000
reg(ps=8) psnr(dB): 26.512973, mse: 0.002232, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 32.955246, mse: 0.000506, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 34.258203, mse: 0.000375, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 33.384481, mse: 0.000459, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 26.888912, mse: 0.002047, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 27.970745, mse: 0.001596, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 33.360090, mse: 0.000461, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 23.850446, mse: 0.004121, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 32.402811, mse: 0.000575, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 32.290580, mse: 0.000590, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 29.248747, mse: 0.001189, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 33.204410, mse: 0.000478, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 23.996103, mse: 0.003985, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 29.431572, mse: 0.001140, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 32.755383, mse: 0.000530, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 31.279616, mse: 0.000745, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 31.949461, mse: 0.000638, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 28.063460, mse: 0.001562, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 28.263602, mse: 0.001492, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 31.644491, mse: 0.000685, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 28.615758, mse: 0.001375, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 30.409756, mse: 0.000910, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 34.624346, mse: 0.000345, pk: 1.000000
src_peak: 1.000000
reg(ps=8) psnr(dB): 26.851486, mse: 0.002065, pk: 1.000000
reg(ps=8) sum_mse: 0.030100














