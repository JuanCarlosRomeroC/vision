Barak Ugav 	318336229
Yishai Gronich 	208989186

Setting Up the Environment
==========================

1. Download Python version 2.7 with OpenCV and Numpy configured.
	* We used Anaconda installation at: https://www.continuum.io/downloads
	* Run the following command to make sure the configuration was successful:
		python -c "import cv2"
	  If there are no errors, Python was configured successfully.
	* A common error is caused by having more than one Python installation on the computer. 
	  In this case, you need to precisely specify the python.exe file you want to use, 
	  i.e. C:\Anaconda\python.exe, instead of just running "python"
2. Enter the command line and navigate to the folder with `__main__.py` and `segmentation.py` files.
3. Run the program like so:
	python . <input-image-path> <path-for-output-segmented-image> <path-for-output-mask>

	* More information about running the program - in INSTRUCTIONS.txt
