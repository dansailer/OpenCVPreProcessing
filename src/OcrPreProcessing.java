import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

/**
 * @author dansailer
 *
 */
public class OcrPreProcessing {

	public static final Logger LOGGER = Logger.getLogger(OcrPreProcessing.class.getName());
	/**
	 * Size used for edge detection
	 */
	private static final Size EDGESIZE = new Size(640, 480);
	/**
	 * How many contours should be analyzed?
	 */
	private static final int NUMBER_OF_LARGE_CONTOURS = 5;
	/**
	 * The size of the black border added incase of paper clipping
	 */
	private static final int BORDER_SIZE = 15;
	/**
	 * Fraction of the original picture that a cropped page should have minimum.
	 * Safeguard against wrong crops.
	 */
	public static final double MIN_PAGE_FRACTION = 0.5;
	/**
	 * Enable debug image output
	 */
	public static boolean DEBUG = Boolean.parseBoolean(System.getProperty("DEBUG", "false"));

	/**
	 * Pre processes the image to increase OCR results, by gray scaling and applying
	 * thresholds.
	 * 
	 * @param source
	 *            The source image to prepare for OCR.
	 * @param equalizeHist
	 *            Should the histogram be equalized?
	 * @param blend
	 *            Should the resulting image be blended with the original?
	 * @return Copy of the source image prepared for OCR.
	 */
	public static Mat prepare(Mat source, boolean equalizeHist, boolean blend) {
		LOGGER.log(Level.FINER, "Prepare image for OCR");
		Mat output = new Mat(source.rows(), source.cols(), CvType.CV_8U);

		try {
			// Step: Gray Scale
			LOGGER.log(Level.FINER, "GrayScale");
			Mat gray = new Mat(source.rows(), source.cols(), CvType.CV_8U);
			if (source.channels() >= 3) {
				Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
				gray.copyTo(output);
			} else {
				output.copyTo(gray);
			}

			// Step: Equalize Histogram after gray scaling
			if (equalizeHist) {
				LOGGER.log(Level.FINER, "Equalize Histogram after gray scaling");
				Imgproc.equalizeHist(gray, gray);
			}

			// Step: BilateralFilter blurring for better keeping edges than GaussianBlur
			LOGGER.log(Level.FINER, "BilateralFilter Blur");
			Mat filterSource = new Mat(source.rows(), source.cols(), CvType.CV_8U);
			output.copyTo(filterSource);
			// What are the best parameters for d and sigmaColor and sigmaSpace? 9,300,300 /
			// 7,75,75 ... Trial and error...
			Imgproc.bilateralFilter(filterSource, output, 7, 75, 75);
			filterSource.release();

			// Step: Adaptive Threshold
			LOGGER.log(Level.FINER, "Adaptive Threshold");
			// What are good values for blockSize and c? 11,4 / 11,2 / 13,4 ... trial and
			// error...
			Imgproc.adaptiveThreshold(output, output, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,
					13, 4);

			// Step: Erode to make the lines bigger.
			// Reduces readability of thin text --> disable
			// LOGGER.log(Level.FINER, "Erode to make the threshold lines bigger");
			// Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,
			// 3));
			// Imgproc.erode(output, output, element);
			// element.release();

			// Step: Blend both images together
			if (blend) {
				LOGGER.log(Level.FINER, "Blending both images together");
				Core.addWeighted(output, 0.6, gray, 0.4, 0.0, output);
				gray.release();
			}
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "Preparing image for OCR failed with \"{0}\"", exc.getMessage());
			source.copyTo(output);
		}

		return output;
	}

	/**
	 * Trying to detect a paper page in the given image. Key is to use blurring and
	 * most importantly resizing to remove details so that the Canny Edge algorithm
	 * finds only the useful contours. To improve detection rate, a border is added
	 * in case the page is clipped in the image. Based on Python code from
	 * pyimagesearch:
	 * https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	 * https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	 * 
	 * @param source
	 *            The image to detect the page in.
	 * @param minimalPageSizeOfOriginal
	 *            The fraction of the original image size that the found page size
	 *            should minimally have. This is a safeguard against wrong cropping
	 *            when the actual page size couldn't be found and then only a small
	 *            box of text in the original page is found.
	 * @return The four corners of the page
	 */
	public static ArrayList<Point> detectPage(Mat source, double minimalPageSizeOfOriginal) {
		LOGGER.log(Level.FINE, "Detect paper page borders");
		ArrayList<Point> scaledCorners = new ArrayList<Point>();
		MatOfPoint2f pageContour = new MatOfPoint2f();
		double ratio = 1.0;

		try {
			Mat working = new Mat(source.rows(), source.cols(), source.type());
			source.copyTo(working);

			// Step: Gray Scale
			LOGGER.log(Level.FINER, "GrayScale");
			if (source.channels() >= 3) {
				Imgproc.cvtColor(source, working, Imgproc.COLOR_BGR2GRAY);
			}

			// Step: BilateralFilter blurring for better keeping edges than GaussianBlur
			LOGGER.log(Level.FINER, "BilateralFilter Blur");
			Mat filterSource = new Mat(working.rows(), working.cols(), CvType.CV_8U);
			working.copyTo(filterSource);
			Imgproc.bilateralFilter(filterSource, working, 7, 75, 75);
			filterSource.release();

			// Step: Get ratio and size for better edge detection
			LOGGER.log(Level.FINER, "Resizing...");
			ratio = Math.min(EDGESIZE.width / source.width(), EDGESIZE.height / source.height());
			Size newSize = new Size(source.width() * ratio, source.height() * ratio);
			Imgproc.resize(working, working, newSize);

			// Step: Add black border in case page is clipped
			LOGGER.log(Level.FINER, "Add black border");
			Core.copyMakeBorder(working, working, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
					Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

			// Step: GaussianBlur
			LOGGER.log(Level.FINER, "GaussianBlur");
			Imgproc.GaussianBlur(working, working, new Size(5, 5), 0.0);

//			// Step: Dilating to remove uncessary data
//			Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
//			Imgproc.dilate(working, working, kernel);

			// Step: Canny
			LOGGER.log(Level.FINER, "Canny edge detection");
			Mat edges = new Mat(working.rows(), working.cols(), working.type());
			// Calculate auto thresholds
			// https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
			double median = median(working);
			int lower = (int) Math.max(0, (1.0 - 0.33) * median);
			int upper = (int) Math.max(255, (1.0 + 0.33) * median);
			Imgproc.Canny(working, edges, lower, upper, 3, true);
			// Imgproc.Canny(working, edges, 75, 200, 3, true);
			if (DEBUG) {
				HighGui.imshow("Canny", edges);
				Imgproc.cvtColor(working, working, Imgproc.COLOR_GRAY2BGR);
			}

			// Step: Find contours, sort and take the largest couple
			LOGGER.log(Level.FINER, "Find contours, sort and take the largest couple");
			ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat hierarchy = new Mat();
			// Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST,
			// Imgproc.CHAIN_APPROX_SIMPLE);
			//https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
			Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
			hierarchy.release();

			// Close and approximate contours
			ArrayList<MatOfPoint> approximate = new ArrayList<MatOfPoint>(contours.size());
//			ArrayList<MatOfPoint> hull = new ArrayList<MatOfPoint>(contours.size());
			for (MatOfPoint cnt : contours) {
				// approximate the contour
				MatOfPoint2f mop2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(mop2f, true) * 0.01; //* 0.02;
				MatOfPoint mop = new MatOfPoint();
				Imgproc.approxPolyDP(mop2f, mop2f, epsilon, true);
				mop2f.convertTo(mop, CvType.CV_32S);
				approximate.add(mop);
				
//				// close contour by laying hull around contours
//				MatOfInt moi = new MatOfInt();
//				Imgproc.convexHull(mop, moi, false);
//				mop.create((int) moi.size().height, 1, CvType.CV_32SC2);
//				for (int j = 0; j < moi.size().height; j++) {
//					int index = (int) moi.get(j, 0)[0];
//					double[] point = new double[] { cnt.get(index, 0)[0], cnt.get(index, 0)[1] };
//					mop.put(j, 0, point);
//				}
//				hull.add(mop);
			}

			// Only check to largest couple of contours
			int contourCount = Math.min(contours.size(), NUMBER_OF_LARGE_CONTOURS);
			LOGGER.log(Level.FINER, "Number of contours: {0}", contours.size());
			approximate.sort(new ContourComparator());
			List<MatOfPoint> largeContours = approximate.subList(0, contourCount);
			if (DEBUG) {
				ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
				if (contourCount > 0) {
					Point[] test = approximate.get(0).toArray();
					LOGGER.log(Level.INFO, "Red contour: {0} {1} {2} {3} {4} {5} {6}", test);
					temp.add(largeContours.get(0));
					Imgproc.drawContours(working, temp, -1, new Scalar(255, 0, 0), 2);
					temp.clear();
				}
				if (contourCount > 1) {
					Point[] test = approximate.get(1).toArray();
					LOGGER.log(Level.INFO, "Green contour: {0} {1} {2} {3} {4} {5} {6}", test);
					temp.add(largeContours.get(1));
					Imgproc.drawContours(working, temp, -1, new Scalar(0, 255, 0), 2);
					temp.clear();
				}
				if (contourCount > 2) {
					Point[] test = approximate.get(2).toArray();
					LOGGER.log(Level.INFO, "Blue contour: {0} {1} {2} {3} {4} {5} {6}", test);
					temp.add(largeContours.get(2));
					Imgproc.drawContours(working, temp, -1, new Scalar(0, 0, 255), 2);
					temp.clear();
				}
				if (contourCount > 3) {
					Point[] test = approximate.get(3).toArray();
					LOGGER.log(Level.INFO, "Cyan contour: {0} {1} {2} {3} {4} {5} {6}", test);
					temp.add(largeContours.get(3));
					Imgproc.drawContours(working, temp, -1, new Scalar(255, 255, 0), 2);
					
				}
			}

			// Step: Loop over and find 4 cornered approximated convex contour searching
			// from large to smaller
			LOGGER.log(Level.FINER, "Loop over and find 4 cornered convex contour searching from large to smaller");
			// Give a threshold of additional 2pixels per border around image
			long imageSize = (edges.rows() - 2 * BORDER_SIZE - 4) * (edges.cols() - 2 * BORDER_SIZE - 4);
			for (MatOfPoint cnt : largeContours) {
				// approximate the contour
				MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(c2f, true);
				MatOfPoint2f approx2f = new MatOfPoint2f();
				MatOfPoint approx = new MatOfPoint();
				Imgproc.approxPolyDP(c2f, approx2f, 0.2 * epsilon, true);
				approx2f.convertTo(approx, CvType.CV_32S);
				boolean convex = Imgproc.isContourConvex(cnt);
				double area = Imgproc.contourArea(cnt);
				LOGGER.log(Level.FINER,
						"Approximated contour - Total: {0} Convex: {1} Continuous: {2} Areasize: {3} ImageSize: {4}",
						new Object[] { cnt.total(), convex, cnt.isContinuous(), area, imageSize });
				// contour is too small --> exit, since the contours are sorted by area size.
				if (imageSize * minimalPageSizeOfOriginal > area) {
					if (DEBUG) {
						HighGui.imshow("Page", working);
					}
					LOGGER.log(Level.INFO,
							"Found page size is too small compared to the original image size {0} / {1}.",
							new Object[] { area, imageSize });
					break;
				}
				// contour is image border (from adding borders) --> ignore
				if (area > imageSize) {
					if (DEBUG) {
						HighGui.imshow("Page", working);
					}
					LOGGER.log(Level.INFO, "Size is too big to be a real crop benefit: {0} / {1}.",
							new Object[] { area, imageSize });
					continue;
				}
				// approximated contour has four points and is convex --> page found
				if (cnt.total() == 4 && convex) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Imgproc.drawContours(working, temp, -1, new Scalar(255, 255, 255), 5);
						HighGui.imshow("Page", working);
					}
					pageContour = approx2f;
					break;
				}
			}
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "Finding page failed with \"{0}\"", exc.getMessage());
		}

		// Step: If page found, scale and calculate corners
		LOGGER.log(Level.FINER, "If page found, scale and calculate corners");
		if (!pageContour.empty()) {
			Point[] corners = pageContour.toArray();
			if (DEBUG) {
				Mat cropped = new Mat();
				ArrayList<Point> points = new ArrayList<Point>();
				points.addAll(pageContour.toList());
				cropped = transform(cropped, points);
				HighGui.imshow("Cropped", cropped);
			}
			for (Point p : corners) {
				scaledCorners.add(
						new Point(Math.round((p.x - BORDER_SIZE) / ratio), Math.round((p.y - BORDER_SIZE) / ratio)));
			}
		} else {
			scaledCorners.add(new Point(0, 0));
			scaledCorners.add(new Point(source.cols(), 0));
			scaledCorners.add(new Point(0, source.rows()));
			scaledCorners.add(new Point(source.cols(), source.rows()));
		}
		if (DEBUG) {
			HighGui.waitKey();
		}
		LOGGER.log(Level.FINE, "Found these corners: {0}, {1}, {2}, {3}", scaledCorners.toArray());
		return scaledCorners;
	}

	/**
	 * Creates a copy of the given image and does a perspective transformation of
	 * the given quadrilateral. The quadrilateral will be warped into a rectangle.
	 * Based on Python code from pyimagesearch:
	 * https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	 * 
	 * @param source
	 *            The source image
	 * @param points
	 *            4 points indicating corners
	 * @return The new cropped and transformed image.
	 */
	public static Mat transform(Mat source, ArrayList<Point> points) {
		LOGGER.log(Level.FINE, "Transforming image with corners {0}, {1}, {2}, {3}", points.toArray());
		Mat result = new Mat(source.rows(), source.cols(), source.type());
		source.copyTo(result);

		// Consistent ordering of points: tl, tr, br, bl
		Point[] sortedCorners = orderPoints(points);
		if (sortedCorners == null) {
			LOGGER.log(Level.WARNING, "Point array has {0} instead of 4 points as corners.", points.size());
			return result;
		}

		try {
			Point tl = sortedCorners[0];
			Point tr = sortedCorners[1];
			Point br = sortedCorners[2];
			Point bl = sortedCorners[3];

			// Width of the new image is maximum distance of (br,bl) and or (tr,tl)
			double widthBottom = Math.sqrt(Math.pow((br.x - bl.x), 2) + Math.pow((br.y - bl.y), 2));
			double widthTop = Math.sqrt(Math.pow((tr.x - tl.x), 2) + Math.pow((tr.y - tl.y), 2));
			double width = Math.max(widthBottom, widthTop);

			// Height of the new image is maximum distance of (tr,br) and or (tl,bl)
			double heightRight = Math.sqrt(Math.pow((tr.x - br.x), 2) + Math.pow((tr.y - br.y), 2));
			double heightLeft = Math.sqrt(Math.pow((tl.x - bl.x), 2) + Math.pow((tl.y - bl.y), 2));
			double height = Math.max(heightRight, heightLeft);
			LOGGER.log(Level.FINER, "New dimensions: {0} x {1}", new Object[] { width, height });

			// Construct set of destination points to obtain a "birds eye view"
			// from width and height
			MatOfPoint2f sourceCorners = new MatOfPoint2f();
			sourceCorners.fromArray(sortedCorners);
			Point[] sortedDestinationCorners = new Point[4];
			sortedDestinationCorners[0] = new Point(0, 0);
			sortedDestinationCorners[1] = new Point(width - 1, 0);
			sortedDestinationCorners[2] = new Point(width - 1, height - 1);
			sortedDestinationCorners[3] = new Point(0, height - 1);
			MatOfPoint2f destinationCorners = new MatOfPoint2f();
			destinationCorners.fromArray(sortedDestinationCorners);
			LOGGER.log(Level.FINER, "Old: {0} {1} {2} {3}", sortedCorners);
			LOGGER.log(Level.FINER, "New: {0} {1} {2} {3}", sortedDestinationCorners);

			// Crop and transform
			Mat M = Imgproc.getPerspectiveTransform(sourceCorners, destinationCorners);
			Imgproc.warpPerspective(result, result, M, new Size(width, height));
			M.release();
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "Transformation failed with \"{0}\"", exc.getMessage());
		}

		return result;
	}

	/**
	 * Sorts 4 points into the following order: TopLeft TopRight BottomRight
	 * BottomLeft Based on Python code from pyimagesearch:
	 * https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	 * 
	 * @param points
	 *            Array list of exactly 4 points
	 * @return Point araray with the points sorted
	 */
	public static Point[] orderPoints(ArrayList<Point> points) {
		if (points.size() != 4) {
			return null;
		}
		Point[] result = new Point[4];
		// Sorting by sum of x+y
		points.sort(new CoordinateSumComparator());
		// Top Left Point has smallest x+y sum
		result[0] = points.get(0);
		// Bottom Right Point has largest x+y sum
		result[2] = points.get(3);
		// Sorting by difference x-y
		points.sort(new CoordinateDifferenceComparator());
		// Top Right Point has the smallest x-y difference
		result[1] = points.get(0);
		// Bottom Left Point has the largest x-y difference
		result[3] = points.get(3);
		return result;
	}

	// https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp

	public static double median(Mat image) {
		double m = (image.rows() * image.cols()) / 2;
		double median = -1;
		int sizeOfHist = 256;
		MatOfInt histSize = new MatOfInt(sizeOfHist);
		MatOfFloat range = new MatOfFloat(0.0f, 256.0f);
		MatOfInt channels = new MatOfInt(0);
		Boolean accumulate = false;
		Mat hist = new Mat();

		ArrayList<Mat> images = new ArrayList<Mat>();
		images.add(image);
		Imgproc.calcHist(images, channels, new Mat(), hist, histSize, range, accumulate);

		long bin = 0;
		for (int i = 0; i < 256 && median < 0; i++) {
			bin = bin + Math.round(hist.get(i, 0)[0]);
			if (bin > m && median < 0)
				median = i;
		}

		return median;

	}

	// http://cartucho.github.io/tutorial_histogram_calculation.html
	public static Mat calculateHistogram(Mat image) {
		List<Mat> gray_planes = new ArrayList<Mat>();
		gray_planes.add(image);

		int sizeOfHist = 256;
		MatOfInt histSize = new MatOfInt(sizeOfHist);
		MatOfFloat range = new MatOfFloat(0.0f, 256.0f);
		MatOfInt channels = new MatOfInt(0);
		Boolean accumulate = false;
		Mat hist_gray = new Mat();

		Imgproc.calcHist(gray_planes, channels, new Mat(), hist_gray, histSize, range, accumulate);

		// Draw the histogram
		int hist_w = 512;
		int hist_h = 400;
		int bin_w = (int) Math.round((double) hist_w / histSize.get(0, 0)[0]);
		Mat histImage = new Mat(hist_h, hist_w, CvType.CV_8UC3, new Scalar(0, 0, 0));

		Core.normalize(hist_gray, hist_gray, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
		for (int i = 1; i < sizeOfHist; i++) {
			Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_gray.get(i - 1, 0)[0])),
					new Point(bin_w * (i), hist_h - Math.round(hist_gray.get(i, 0)[0])), new Scalar(255, 255, 255), 2,
					8, 0);
		}

		return histImage;
	}

}
