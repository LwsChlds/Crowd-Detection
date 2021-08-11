# Deepstream inferencing

> using the [NVIDIA Deepstream SDK](https://developer.nvidia.com/deepstream-sdk) to run infernecing on an onnx model/engine file based of the [deepstream python apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) examples

Prequisites:
- DeepStreamSDK 5.1
- Python 3.6
- Gst-python
- NumPy package
- OpenCV package

To run the app:

    python3 deepstream-crowd-detection.py <config_file> <jpeg/mjpeg stream>

For example:

    python3 deepstream-crowd-detection.py crowd_detector.txt example.mjpeg.avi

## Issues with code
> the deep stream code was not completed as a detection accuracy problem occurred

The issue occurred due to the differences in preprocessing done by PyTorch and deepsteam

In pyTorch the equation used is:

     y = (x - mean) / std

And in deepstream it is:

    y' = net-scale-factor * ( x' - mean')

The mean and std used in PyTorch are three values corresponding to the RGB values of the input image resulting in channel-wise normalisation.

Unfortunately, deepstream does not currently support channel-wise normalisation. As a result, in the [config settings of deepstream](crowd_detector.txt), you can specify three values for offsets(mean); however, the value corresponding to the std is net-scale-factor which can only be a singular value.

Additionally, the model is trained on input values between 0 and 1, which can be obtained by dividing the original RGB values by 225.

As a result, rearranging the deepstream equation into the format of the PyTorch equation gives the following:

    1 / (255 * 255 * STD) *  (x - 255(mean)) = net-scale-factor * (x - offsets)

As a result, to get the same output net-scale-factor must be set to:

    1 / (255 * 255 * STD)

And offsets must be set to:

    255(mean)

However, as only one net-scale-factor can be set, you would have to take an average of the three. Usually, this would only be a slight difference; however, as this value is multiplied by 255 twice, the small change is magnified to have a more significant difference resulting in input values that the model cannot detect.


