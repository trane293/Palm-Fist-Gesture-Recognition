#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
int main() {

	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "Could not open Camera stream!";
		return -1;
	}

	// Load open_palm cascade and eye cascade files (.xml file) The XML files should be
	// in the same directory as the project.

	CascadeClassifier open_palm_cascade;
	CascadeClassifier closed_palm_cascade;
	open_palm_cascade.load("open_palm.xml");
	closed_palm_cascade.load("closed_palm.xml");

	if (open_palm_cascade.empty()) {
		cout << "Could not load open_palm configuration file! "
			"Check directory! " << endl << "Press Q to Quit!" << endl;
		while (char(waitKey(0)) != 'q') {}
		return -2;
	}

	if (closed_palm_cascade.empty()) {
		cout << "Could not load closed_palm configuration file! "
			"Check directory! " << endl << "Press Q to Quit!" << endl;
		while (char(waitKey(0)) != 'q') {}
		return -2;
	}

	// Start the open_palm and eye detection phase
	vector<Rect> open_palms;
	vector<Rect> closed_palms;

	while (char(waitKey(1)) != 'q' && cap.isOpened()) {
		double t0 = getTickCount();
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			cout << "Video over!";
			break;
		}

		// Detect all the open_palms.
		open_palm_cascade.detectMultiScale(frame, open_palms, 1.3, 4, 0, Size(50, 50));
		closed_palm_cascade.detectMultiScale(frame, closed_palms, 1.3, 4, 0, Size(50, 50));
		
		if (closed_palms.size() != 0) {
		
			for (int i = 0; i < closed_palms.size(); i++) {

				cout << "=============Detected a closed_palm!=============" << endl;

				// Top left and bottom right points of rectangle.
				Point closed_palm_rect_p1(closed_palms[i].x, closed_palms[i].y);
				Point closed_palm_rect_p2(closed_palms[i].x + closed_palms[i].width, closed_palms[i].y + closed_palms[i].height);

				// Draw the rectangle in the image.
				rectangle(frame, closed_palm_rect_p1, closed_palm_rect_p2, Scalar(0, 255, 0));
				putText(frame, "Closed Palm", closed_palm_rect_p1, FONT_HERSHEY_SIMPLEX,
					1, Scalar(0, 255, 0), 1, 5, false);
			}
		}
		for (int i = 0; i < open_palms.size(); i++) {

			cout << "=============Detected an open_palm!=============" << endl;

			// Top left and bottom right points of rectangle.
			Point open_palm_rect_p1(open_palms[i].x, open_palms[i].y);
			Point open_palm_rect_p2(open_palms[i].x + open_palms[i].width, open_palms[i].y + open_palms[i].height);

			// Draw the rectangle in the image.
			rectangle(frame, open_palm_rect_p1, open_palm_rect_p2, Scalar(255, 0, 0));
			putText(frame, "Open Palm", open_palm_rect_p1, FONT_HERSHEY_SIMPLEX,
				1, Scalar(255, 0, 0), 1, 5, false);
		}

		imshow("Video Capture", frame);
		cout << "Frame rate = " << getTickFrequency() / (getTickCount() - t0) << endl;
	}
	return 0;
}
