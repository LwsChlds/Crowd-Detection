## Converting pytorch model into the onnx format

The pytoch model being used is an adapted version of [CrowdCounting on VisDrone2020](https://github.com/pasqualedem/CrowdCounting-on-VisDrone2020) created by pasqualedem and uses the [MobileCount](https://github.com/SelinaFelton/MobileCount) models plus 2 two variants of it.

### Running the code

Install [requirement.txt](requirement.txt) and Python 3.8

To execute from the src file use:
  
    python main.py args
    
The first arg is the modality, can be: run or onnx

For run mode, you must also specify:

<ul>
<li>--path: path to the video or image file, or the folder containing the image</li>
<li>--callbacks: list of callback function to be executed after the forward of each element</li>
</ul>

For example to count the amount of pedestrians on the test image use:

    python main.py run --path crowd.jpg --callbacks [\'count_callback\']

To create the onnx model use:

    python main.py onnx

