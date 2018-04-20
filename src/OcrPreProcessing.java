import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
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
	 * The minimal threshold of variance of laplacian before considering a image
	 * sharp.
	 */
	public static final double MIN_VARIANCE_OF_LAPLACIAN = 70;
	/**
	 * The minimal threshold of modified laplacian before considering a image sharp.
	 */
	public static final double MIN_MODIFIED_LAPLACIAN = 4;
	/**
	 * Enable debug image output
	 */
	public static boolean DEBUG = Boolean.parseBoolean(System.getProperty("DEBUG", "false"));

	/**
	 * Returns true if the image is probably blurry when variance of laplacian and
	 * modified laplacian are above the threshold.
	 * 
	 * @param source
	 *            image to calculate blurriness
	 * @return is blurry or not
	 */
	public static boolean isBlurry(Mat source) {
		double blurSrc = varianceOfLaplacian(source);
		double modifiedSrc = modifiedLaplacian(source);
		if (blurSrc < MIN_VARIANCE_OF_LAPLACIAN && modifiedSrc < MIN_MODIFIED_LAPLACIAN) {
			LOGGER.log(Level.INFO, "\n BLURRY - varianceOfLaplacian: {0}, modifiedLaplacian: {1}", new Object[] { blurSrc, modifiedSrc });
			return true;
		}
		return false;
	}

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
			// https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
			// http://opencvpython.blogspot.ch/2012/06/sudoku-solver-part-2.html
			Imgproc.adaptiveThreshold(output, output, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 13, 4);

			// Step: Blend both images together
			if (blend) {
				LOGGER.log(Level.FINER, "Blending both images together");
				Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
				Core.addWeighted(output, 0.6, source, 1, 0.0, output);
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
		MatOfPoint pageContour = new MatOfPoint();
		double ratio = 1.0;
		Mat debugImg = null;
		if (DEBUG) {
			debugImg = new Mat();
			source.copyTo(debugImg);
		}

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
			if (DEBUG) {
				Imgproc.resize(debugImg, debugImg, newSize);
			}

			// Step: Add black border in case page is clipped
			LOGGER.log(Level.FINER, "Add black border");
			Core.copyMakeBorder(working, working, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
			if (DEBUG) {
				Core.copyMakeBorder(debugImg, debugImg, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT,
						new Scalar(0, 0, 0));
			}

			// Step: GaussianBlur
			LOGGER.log(Level.FINER, "GaussianBlur");
			Imgproc.GaussianBlur(working, working, new Size(5, 5), 0.0);

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

			// Initial Variant
			// Step: Find contours, sort and take the largest couple
			LOGGER.log(Level.FINER, "Find contours, sort and take the largest couple");
			ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat hierarchy = new Mat();
			Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
			hierarchy.release();
			int largeNumber = Math.min(contours.size(), NUMBER_OF_LARGE_CONTOURS);
			LOGGER.log(Level.FINER, "Number of contours: {0}", contours.size());
			contours.sort(new ContourComparator());
			List<MatOfPoint> largeContours = contours.subList(0, largeNumber);
			if (DEBUG) {
				ArrayList<Scalar> colors = new ArrayList<Scalar>();
				ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
				colors.add(new Scalar(255, 55, 55));
				colors.add(new Scalar(0, 255, 0));
				colors.add(new Scalar(0, 0, 255));
				colors.add(new Scalar(255, 255, 0));
				colors.add(new Scalar(0, 255, 255));
				double blurVar = varianceOfLaplacian(source);
				double blurMod = modifiedLaplacian(source);
				String blur = "";
				if (blurVar < MIN_VARIANCE_OF_LAPLACIAN && blurMod < MIN_MODIFIED_LAPLACIAN) {
					blur = String.format("BLURRED (%1$.3f?/%2$.3f)", blurVar, blurMod);
				}
				Imgproc.putText(working, "Normal area: " + blur, new Point(3 * BORDER_SIZE, 3 * BORDER_SIZE), Core.FONT_HERSHEY_PLAIN, 1,
						new Scalar(255, 255, 255), 2);
				for (int i = 0; i < largeNumber; i++) {
					double area = Imgproc.contourArea(largeContours.get(i));
					Imgproc.putText(working, Double.toString(area), new Point(530, working.height() - (i + 3) * 2 * BORDER_SIZE),
							Core.FONT_HERSHEY_PLAIN, 1.5, colors.get(i), 2);
					temp.add(largeContours.get(i));
					Imgproc.drawContours(working, temp, -1, colors.get(i), 2);
					temp.clear();
				}
				HighGui.imshow("Contours", working);
			}

			// Step: Loop over and find 4 cornered approximated convex contour searching
			// from large to smaller
			LOGGER.log(Level.FINER, "Loop over and find 4 cornered convex contour searching from large to smaller");
			long imageSize = (edges.rows() - 2 * BORDER_SIZE - 4) * (edges.cols() - 2 * BORDER_SIZE - 4);
			for (MatOfPoint cnt : largeContours) {
				// approximate the contour
				MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(c2f, true);
				MatOfPoint2f approx2f = new MatOfPoint2f();
				MatOfPoint approx = new MatOfPoint();
				Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
				approx2f.convertTo(approx, CvType.CV_32S);
				c2f.release();

				// Initial Variant
				double area = Imgproc.contourArea(approx2f);
				boolean convex = Imgproc.isContourConvex(approx);
				LOGGER.log(Level.FINER, "Approximated contour - Total: {0} Convex: {1} Continuous: {2} Areasize: {3} ImageSize: {4}",
						new Object[] { cnt.total(), convex, cnt.isContinuous(), area, imageSize });
				// contour is too small --> exit, since the contours are sorted by area size.
				if (imageSize * minimalPageSizeOfOriginal > area) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(255, 0, 0), 5);
						HighGui.imshow("Approx", tempImg);
						tempImg.release();
					}
					LOGGER.log(Level.INFO, "Found page size is too small compared to the original image size {0}/{1}.",
							new Object[] { area, imageSize });
					approx2f.release();
					approx.release();
					break;
				}
				// contour is image border (from adding borders) --> ignore
				if (area > imageSize) {
					LOGGER.log(Level.INFO, "Size is too big to be a real crop benefit: {0} / {1}.", new Object[] { area, imageSize });
					continue;
				}
				// approximated contour has four points and is convex --> page found
				if (approx.total() == 4 && convex) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(0, 255, 0), 5);
						HighGui.imshow("Approx", tempImg);
					}
					pageContour = approx;
					approx2f.release();
					break;
				} else {
					LOGGER.log(Level.INFO, "Area has not four points or is not convex - Pts: {0} Convex: {1} Area: {2}",
							new Object[] { approx.total(), convex, area });
				}
				approx2f.release();
				approx.release();
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
				cropped = transform(debugImg, points);
				HighGui.imshow("Cropped", cropped);
			}
			for (Point p : corners) {
				scaledCorners.add(new Point(Math.round((p.x - BORDER_SIZE) / ratio), Math.round((p.y - BORDER_SIZE) / ratio)));
			}
		} else {
			if (DEBUG) {
				HighGui.imshow("Cropped", debugImg);
			}
			scaledCorners.add(new Point(0, 0));
			scaledCorners.add(new Point(source.cols(), 0));
			scaledCorners.add(new Point(0, source.rows()));
			scaledCorners.add(new Point(source.cols(), source.rows()));
		}

		if (DEBUG) {
			SetDebugWindows();
		}
		LOGGER.log(Level.FINE, "Found these corners: {0}, {1}, {2}, {3}", scaledCorners.toArray());
		return scaledCorners;
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
	public static ArrayList<Point> detectPageWithMinRect(Mat source, double minimalPageSizeOfOriginal) {
		LOGGER.log(Level.FINE, "Detect paper page borders");
		ArrayList<Point> scaledCorners = new ArrayList<Point>();
		MatOfPoint pageContour = new MatOfPoint();
		double ratio = 1.0;
		Mat debugImg = null;
		if (DEBUG) {
			debugImg = new Mat();
			source.copyTo(debugImg);
		}

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
			if (DEBUG) {
				Imgproc.resize(debugImg, debugImg, newSize);
			}

			// Step: Add black border in case page is clipped
			LOGGER.log(Level.FINER, "Add black border");
			Core.copyMakeBorder(working, working, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
			if (DEBUG) {
				Core.copyMakeBorder(debugImg, debugImg, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT,
						new Scalar(0, 0, 0));
			}

			// Step: GaussianBlur
			LOGGER.log(Level.FINER, "GaussianBlur");
			Imgproc.GaussianBlur(working, working, new Size(5, 5), 0.0);

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

			// Sort by MinRect
			// Step: Find contours, sort and take the largest couple
			LOGGER.log(Level.FINER, "Find contours, sort and take the largest couple");
			ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat hierarchy = new Mat();
			Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
			hierarchy.release();
			int largeNumber = Math.min(contours.size(), NUMBER_OF_LARGE_CONTOURS);
			LOGGER.log(Level.FINER, "Number of contours: {0}", contours.size());
			contours.sort(new ContourComparatorMinRect());
			List<MatOfPoint> largeContours = contours.subList(0, largeNumber);
			ArrayList<MatOfPoint> closedContours = new ArrayList<MatOfPoint>();
			for (MatOfPoint cnt : largeContours) {
				if (Imgproc.isContourConvex(cnt)) {
					closedContours.add(cnt);
				} else {
					MatOfPoint2f mop2f = new MatOfPoint2f(cnt.toArray());
					RotatedRect area = Imgproc.minAreaRect(mop2f);
					Point[] pts = new Point[4];
					area.points(pts);
					MatOfPoint rect = new MatOfPoint();
					rect.fromArray(pts);
					closedContours.add(rect);
				}
			}

			if (DEBUG) {
				ArrayList<Scalar> colors = new ArrayList<Scalar>();
				ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
				colors.add(new Scalar(255, 55, 55));
				colors.add(new Scalar(0, 255, 0));
				colors.add(new Scalar(0, 0, 255));
				colors.add(new Scalar(255, 255, 0));
				colors.add(new Scalar(0, 255, 255));
				double blurVar = varianceOfLaplacian(source);
				double blurMod = modifiedLaplacian(source);
				String blur = "";
				if (blurVar < MIN_VARIANCE_OF_LAPLACIAN && blurMod < MIN_MODIFIED_LAPLACIAN) {
					blur = String.format("BLURRED (%1$.3f?/%2$.3f)", blurVar, blurMod);
				}
				Imgproc.putText(working, "Min Rect " + blur, new Point(3 * BORDER_SIZE, 3 * BORDER_SIZE), Core.FONT_HERSHEY_PLAIN, 1,
						new Scalar(255, 255, 255), 2);
				for (int i = 0; i < largeNumber; i++) {
					double area = ContourComparatorMinRect.calculateMinRectArea(largeContours.get(i));
					Imgproc.putText(working, Double.toString(area), new Point(530, working.height() - (i + 3) * 2 * BORDER_SIZE),
							Core.FONT_HERSHEY_PLAIN, 1.5, colors.get(i), 2);
					temp.add(largeContours.get(i));
					Imgproc.drawContours(working, temp, -1, colors.get(i), 2);
					temp.clear();
				}
				HighGui.imshow("Contours", working);
			}

			// Step: Loop over and find 4 cornered approximated convex contour searching
			// from large to smaller
			LOGGER.log(Level.FINER, "Loop over and find 4 cornered convex contour searching from large to smaller");
			long imageSize = (edges.rows() - 2 * BORDER_SIZE - 4) * (edges.cols() - 2 * BORDER_SIZE - 4);
			for (MatOfPoint cnt : closedContours) {
				// approximate the contour
				MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(c2f, true);
				MatOfPoint2f approx2f = new MatOfPoint2f();
				MatOfPoint approx = new MatOfPoint();
				Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
				approx2f.convertTo(approx, CvType.CV_32S);
				double area = Imgproc.contourArea(approx2f);
				boolean convex = Imgproc.isContourConvex(approx);
				LOGGER.log(Level.FINER, "Approximated contour - Total: {0} Convex: {1} Continuous: {2} Areasize: {3} ImageSize: {4}",
						new Object[] { cnt.total(), convex, cnt.isContinuous(), area, imageSize });
				// contour is too small --> exit, since the contours are sorted by area size.
				if (imageSize * minimalPageSizeOfOriginal > area) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(255, 0, 0), 5);
						HighGui.imshow("Approx", tempImg);
					}
					LOGGER.log(Level.INFO, "Found page size is too small compared to the original image size {0}/{1}.",
							new Object[] { area, imageSize });
					break;
				}
				// contour is image border (from adding borders) --> ignore
				if (area > imageSize) {
					LOGGER.log(Level.INFO, "Size is too big to be a real crop benefit: {0} / {1}.", new Object[] { area, imageSize });
					continue;
				}
				// approximated contour has four points and is convex --> page found
				if (approx.total() == 4 && convex) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(0, 255, 0), 5);
						HighGui.imshow("Approx", tempImg);
					}
					pageContour = approx;
					break;
				} else {
					LOGGER.log(Level.INFO, "Area has not four points or is not convex - Pts: {0} Convex: {1} Area: {2}",
							new Object[] { approx.total(), convex, area });
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
				cropped = transform(debugImg, points);
				HighGui.imshow("Cropped", cropped);
			}
			for (Point p : corners) {
				scaledCorners.add(new Point(Math.round((p.x - BORDER_SIZE) / ratio), Math.round((p.y - BORDER_SIZE) / ratio)));
			}
		} else {
			if (DEBUG) {
				HighGui.imshow("Cropped", debugImg);
			}
			scaledCorners.add(new Point(0, 0));
			scaledCorners.add(new Point(source.cols(), 0));
			scaledCorners.add(new Point(0, source.rows()));
			scaledCorners.add(new Point(source.cols(), source.rows()));
		}

		if (DEBUG) {
			SetDebugWindows();
		}
		LOGGER.log(Level.FINE, "Found these corners: {0}, {1}, {2}, {3}", scaledCorners.toArray());
		return scaledCorners;
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
	public static ArrayList<Point> detectPageConvexHull(Mat source, double minimalPageSizeOfOriginal) {
		LOGGER.log(Level.FINE, "Detect paper page borders");
		ArrayList<Point> scaledCorners = new ArrayList<Point>();
		MatOfPoint pageContour = new MatOfPoint();
		double ratio = 1.0;
		Mat debugImg = null;
		if (DEBUG) {
			debugImg = new Mat();
			source.copyTo(debugImg);
		}

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
			if (DEBUG) {
				Imgproc.resize(debugImg, debugImg, newSize);
			}

			// Step: Add black border in case page is clipped
			LOGGER.log(Level.FINER, "Add black border");
			Core.copyMakeBorder(working, working, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
			if (DEBUG) {
				Core.copyMakeBorder(debugImg, debugImg, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, Core.BORDER_CONSTANT,
						new Scalar(0, 0, 0));
			}

			// Step: GaussianBlur
			LOGGER.log(Level.FINER, "GaussianBlur");
			Imgproc.GaussianBlur(working, working, new Size(5, 5), 0.0);

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
			// Close with approx and hull
			// Step: Find contours, close, sort and take the largest couple
			LOGGER.log(Level.FINER, "Find contours, sort and take the largest couple");
			ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat hierarchy = new Mat();
			Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
			hierarchy.release();
			ArrayList<MatOfPoint> closedContours = new ArrayList<MatOfPoint>();
			for (MatOfPoint cnt : contours) {
				// approximate the contour
				// MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				// double epsilon = Imgproc.arcLength(c2f, true);
				// MatOfPoint2f approx2f = new MatOfPoint2f();
				// MatOfPoint approx = new MatOfPoint();
				// Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
				// approx2f.convertTo(approx, CvType.CV_32S);
				// replace cnt with approx below
				if (Imgproc.isContourConvex(cnt)) {
					closedContours.add(cnt);
				} else {
					// create a convex hull around approximation
					MatOfInt moi = new MatOfInt();
					Imgproc.convexHull(cnt, moi, false);
					MatOfPoint mop = new MatOfPoint();
					mop.create((int) moi.size().height, 1, CvType.CV_32SC2);
					for (int j = 0; j < moi.size().height; j++) {
						int index = (int) moi.get(j, 0)[0];
						double[] point = new double[] { cnt.get(index, 0)[0], cnt.get(index, 0)[1] };
						mop.put(j, 0, point);
					}
					closedContours.add(mop);
				}
			}
			int largeNumber = Math.min(closedContours.size(), NUMBER_OF_LARGE_CONTOURS);
			LOGGER.log(Level.FINER, "Number of contours: {0}", closedContours.size());
			closedContours.sort(new ContourComparator());
			List<MatOfPoint> largeContours = closedContours.subList(0, largeNumber);
			if (DEBUG) {
				ArrayList<Scalar> colors = new ArrayList<Scalar>();
				ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
				colors.add(new Scalar(255, 55, 55));
				colors.add(new Scalar(0, 255, 0));
				colors.add(new Scalar(0, 0, 255));
				colors.add(new Scalar(255, 255, 0));
				colors.add(new Scalar(0, 255, 255));
				double blurVar = varianceOfLaplacian(source);
				double blurMod = modifiedLaplacian(source);
				String blur = "";
				if (blurVar < MIN_VARIANCE_OF_LAPLACIAN && blurMod < MIN_MODIFIED_LAPLACIAN) {
					blur = String.format("BLURRED (%1$.3f?/%2$.3f)", blurVar, blurMod);
				}
				Imgproc.putText(working, "Convex Hull " + blur, new Point(3 * BORDER_SIZE, 3 * BORDER_SIZE), Core.FONT_HERSHEY_PLAIN, 1,
						new Scalar(255, 255, 255), 2);
				for (int i = 0; i < largeNumber; i++) {
					double area = Imgproc.contourArea(largeContours.get(i));
					Imgproc.putText(working, Double.toString(area), new Point(530, working.height() - (i + 3) * 2 * BORDER_SIZE),
							Core.FONT_HERSHEY_PLAIN, 1.5, colors.get(i), 2);
					temp.add(largeContours.get(i));
					Imgproc.drawContours(working, temp, -1, colors.get(i), 2);
					temp.clear();
				}
				HighGui.imshow("Contours", working);
			}

			// Step: Loop over and find 4 cornered approximated convex contour searching
			// from large to smaller
			LOGGER.log(Level.FINER, "Loop over and find 4 cornered convex contour searching from large to smaller");
			long imageSize = (edges.rows() - 2 * BORDER_SIZE - 4) * (edges.cols() - 2 * BORDER_SIZE - 4);
			for (MatOfPoint cnt : largeContours) {
				// approximate the contour
				MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(c2f, true);
				MatOfPoint2f approx2f = new MatOfPoint2f();
				MatOfPoint approx = new MatOfPoint();
				Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
				approx2f.convertTo(approx, CvType.CV_32S);
				double area = Imgproc.contourArea(approx);
				boolean convex = Imgproc.isContourConvex(approx);
				LOGGER.log(Level.FINER, "Approximated contour - Total: {0} Convex: {1} Continuous: {2} Areasize: {3} ImageSize: {4}",
						new Object[] { cnt.total(), convex, cnt.isContinuous(), area, imageSize });
				// contour is too small --> exit, since the contours are sorted by area size.
				if (imageSize * minimalPageSizeOfOriginal > area) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(255, 0, 0), 5);
						HighGui.imshow("Approx", tempImg);
					}
					LOGGER.log(Level.INFO, "Found page size is too small compared to the original image size {0}/{1}.",
							new Object[] { area, imageSize });
					break;
				}
				// contour is image border (from adding borders) --> ignore
				if (area > imageSize) {
					LOGGER.log(Level.INFO, "Size is too big to be a real crop benefit: {0} / {1}.", new Object[] { area, imageSize });
					continue;
				}
				// approximated contour has four points and is convex --> page found
				if (approx.total() == 4 && convex) {
					if (DEBUG) {
						ArrayList<MatOfPoint> temp = new ArrayList<MatOfPoint>();
						temp.add(approx);
						Mat tempImg = new Mat();
						debugImg.copyTo(tempImg);
						Imgproc.drawContours(tempImg, temp, -1, new Scalar(0, 255, 0), 5);
						HighGui.imshow("Approx", tempImg);
					}
					pageContour = approx;
					break;
				} else {
					LOGGER.log(Level.INFO, "Area has not four points or is not convex - Pts: {0} Convex: {1} Area: {2}",
							new Object[] { approx.total(), convex, area });
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
				cropped = transform(debugImg, points);
				HighGui.imshow("Cropped", cropped);
			}
			for (Point p : corners) {
				scaledCorners.add(new Point(Math.round((p.x - BORDER_SIZE) / ratio), Math.round((p.y - BORDER_SIZE) / ratio)));
			}
		} else {
			if (DEBUG) {
				HighGui.imshow("Cropped", debugImg);
			}
			scaledCorners.add(new Point(0, 0));
			scaledCorners.add(new Point(source.cols(), 0));
			scaledCorners.add(new Point(0, source.rows()));
			scaledCorners.add(new Point(source.cols(), source.rows()));
		}

		if (DEBUG) {
			SetDebugWindows();
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
			destinationCorners.release();
			sourceCorners.release();
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

	/**
	 * Calculate median for an image.
	 * https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp
	 * 
	 * @param Source
	 *            image
	 * @return Double median
	 */
	public static double median(Mat image) {
		double m = (image.rows() * image.cols()) / 2;
		double median = -1;
		int sizeOfHist = 256;
		MatOfInt histSize = new MatOfInt(sizeOfHist);
		MatOfFloat range = new MatOfFloat(0.0f, 256.0f);
		MatOfInt channels = new MatOfInt(0);
		Boolean accumulate = false;
		Mat hist = new Mat();
		ArrayList<Mat> images = new ArrayList<Mat>(1);
		images.add(image);
		Imgproc.calcHist(images, channels, new Mat(), hist, histSize, range, accumulate);
		long bin = 0;
		for (int i = 0; i < 256 && median < 0; i++) {
			bin = bin + Math.round(hist.get(i, 0)[0]);
			if (bin > m && median < 0)
				median = i;
		}
		histSize.release();
		range.release();
		channels.release();
		hist.release();
		images.clear();
		return median;
	}

	/**
	 * Simplest Color Balance. Performs color balancing via histogram normalization.
	 *
	 * @param img
	 *            input color or gray scale image
	 * @param percent
	 *            controls the percentage of pixels to clip to white and black.
	 *            (normally, choose 1~10)
	 * @return Balanced image in CvType.CV_32F
	 */
	public static Mat SimplestColorBalance(Mat img, int percent) {
		if (percent <= 0)
			percent = 5;
		img.convertTo(img, CvType.CV_8U);
		Mat debugImg = new Mat();
		double ratio = Math.min(EDGESIZE.width / img.width(), EDGESIZE.height / img.height());
		Size newSize = new Size(img.width() * ratio, img.height() * ratio);
		Imgproc.resize(img, debugImg, newSize);
		if (DEBUG) {
			HighGui.namedWindow("Source", HighGui.WINDOW_AUTOSIZE);
			HighGui.imshow("Source", debugImg);
		}
		List<Mat> channels = new ArrayList<>();
		int rows = img.rows(); // number of rows of image
		int cols = img.cols(); // number of columns of image
		int chnls = img.channels(); // number of channels of image
		double halfPercent = percent / 200.0;
		if (chnls == 3)
			Core.split(img, channels);
		else
			channels.add(img);
		List<Mat> results = new ArrayList<>();
		for (int i = 0; i < chnls; i++) {
			// find the low and high precentile values (based on the input percentile)
			Mat flat = new Mat();
			channels.get(i).reshape(1, 1).copyTo(flat);
			Core.sort(flat, flat, Core.SORT_ASCENDING);
			double lowVal = flat.get(0, (int) Math.floor(flat.cols() * halfPercent))[0];
			double topVal = flat.get(0, (int) Math.ceil(flat.cols() * (1.0 - halfPercent)))[0];
			// saturate below the low percentile and above the high percentile
			Mat channel = channels.get(i);
			for (int m = 0; m < rows; m++) {
				for (int n = 0; n < cols; n++) {
					if (channel.get(m, n)[0] < lowVal)
						channel.put(m, n, lowVal);
					if (channel.get(m, n)[0] > topVal)
						channel.put(m, n, topVal);
				}
			}
			Core.normalize(channel, channel, 0.0, 255.0 / 2, Core.NORM_MINMAX);
			channel.convertTo(channel, CvType.CV_32F);
			results.add(channel);
		}
		Mat outval = new Mat(img.rows(), img.cols(), img.type());
		Core.merge(results, outval);
		
		Mat debugResultImg = new Mat();
		LOGGER.log(Level.INFO, "Type: {0}", outval.type());
		Imgproc.resize(outval, debugResultImg, newSize);
		debugResultImg.convertTo(debugResultImg, CvType.CV_8U);
		LOGGER.log(Level.INFO, "Type: {0}", debugResultImg.type());
		if (DEBUG) {
			HighGui.namedWindow("Simplest", HighGui.WINDOW_AUTOSIZE);
			HighGui.imshow("Simplest", debugResultImg);
			HighGui.waitKey();
		}
		return outval;
	}

	/**
	 * This is the color balancing technique used in Adobe Photoshop's "auto levels"
	 * command. The idea is that in a well balanced photo, the brightest color
	 * should be white and the darkest black. Thus, we can remove the color cast
	 * from an image by scaling the histograms of each of the R, G, and B channels
	 * so that they span the complete 0-255 scale. In contrast to the other color
	 * balancing algorithms, this method does not separate the estimation and
	 * adaptation steps. In order to deal with outliers, Simplest Color Balance
	 * saturates a certain percentage of the image's bright pixels to white and dark
	 * pixels to black. The saturation level is an adjustable parameter that affects
	 * the quality of the output. Values around 0.01 are typical.
	 * 
	 * http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
	 * http://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
	 * 
	 * @param src
	 *            input color or gray scale image
	 * @param percent
	 *            controls the percentage of pixels to clip to white and black.
	 *            Values around 0.01 are typical.
	 * @return Balanced image in CvType.CV_32F
	 */
	public static Mat SimplestColorBalancing(Mat src, double percent) {
		src.convertTo(src, CvType.CV_8U);
		Mat debugImg = new Mat();
		double ratio = Math.min(EDGESIZE.width / src.width(), EDGESIZE.height / src.height());
		Size newSize = new Size(src.width() * ratio, src.height() * ratio);
		Imgproc.resize(src, debugImg, newSize);
		if (DEBUG) {
			HighGui.namedWindow("Source", HighGui.WINDOW_AUTOSIZE);
			HighGui.imshow("Source", debugImg);
		}
		
		if (!(percent > 0 && percent < 100)) {
			percent = 0.01f;
		}
		double halfPercent = percent / 200.0f;
		List<Mat> channels = new ArrayList<Mat>();
		if (src.channels() == 3)
			Core.split(src, channels);
		else
			channels.add(src);
		for (Mat channel : channels) {
			// find the low and high percentile values (based on the input percentile)
			Mat flat = new Mat();
			channel.reshape(1, 1).copyTo(flat);
			Core.sort(flat, flat, Core.SORT_EVERY_ROW + Core.SORT_ASCENDING);
			double lowVal = flat.get(0, (int) Math.floor(flat.cols() * halfPercent))[0];
			double topVal = flat.get(0, (int) Math.ceil(flat.cols() * (1.0 - halfPercent)))[0];

			// saturate below the low percentile and above the high percentile
			for (int x = 0; x < src.rows(); x++) {
				for (int y = 0; y < src.cols(); y++) {
					if (channel.get(x, y)[0] < lowVal)
						channel.put(x, y, lowVal);
					if (channel.get(x, y)[0] > topVal)
						channel.put(x, y, topVal);
				}
			}
			channel.setTo(Scalar.all(lowVal));
			//tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
	        //tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

			// scale the channel
			Core.normalize(channel, channel, 0.0, 255.0, Core.NORM_MINMAX);
			channel.convertTo(channel, CvType.CV_32F);
			flat.release();
		}
		Mat result = new Mat();
		Core.merge(channels, result);
		
		Mat debugResultImg = new Mat();
		LOGGER.log(Level.INFO, "Type: {0}", result.type());
		Imgproc.resize(result, debugResultImg, newSize);
		debugResultImg.convertTo(debugResultImg, CvType.CV_8U);
		LOGGER.log(Level.INFO, "Type: {0}", debugResultImg.type());
		if (DEBUG) {
			HighGui.namedWindow("Simplest", HighGui.WINDOW_AUTOSIZE);
			HighGui.imshow("Simplest", debugResultImg);
			HighGui.waitKey();
		}
		
		return result;
	}

	/**
	 * OpenCV port of 'LAPM' algorithm (Nayar89) https://stackoverflow.com/a/7768918
	 * http://radjkarl.github.io/imgProcessor/_modules/imgProcessor/measure/sharpness/parameters.html
	 * 
	 * @param src
	 *            The source image
	 * @return modifiedLaplacian
	 */
	public static double modifiedLaplacian(Mat src) {
		double focusMeasure = 0;
		MatOfDouble M = new MatOfDouble(3, 1, CvType.CV_32F);
		M.put(0, 0, -1, 2, -1);
		Mat G = Imgproc.getGaussianKernel(3, -1, CvType.CV_64F);
		Mat Lx = new Mat();
		Imgproc.sepFilter2D(src, Lx, CvType.CV_64F, M, G);
		Mat Ly = new Mat();
		Imgproc.sepFilter2D(src, Ly, CvType.CV_64F, G, M);
		Mat absLx = new Mat();
		Core.absdiff(Lx, Scalar.all(0), absLx);
		Mat absLy = new Mat();
		Core.absdiff(Ly, Scalar.all(0), absLy);
		Mat FM = new Mat();
		Core.add(absLx, absLy, FM);
		focusMeasure = Core.mean(FM).val[0];
		M.release();
		G.release();
		Lx.release();
		Ly.release();
		absLx.release();
		absLy.release();
		FM.release();
		return focusMeasure;
	}

	/**
	 * OpenCV port of 'LAPV' algorithm (Pech2000)
	 * https://stackoverflow.com/a/7768918
	 * 
	 * This is the best algorith to find blurriness according to
	 * https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
	 * https://stackoverflow.com/questions/36413394/opencv-variation-of-the-laplacian-java
	 * 
	 * He is using 100 as a threshold
	 * 
	 * @param src
	 *            The source image
	 * @return varianceOfLaplacian
	 */
	public static double varianceOfLaplacian(Mat src) {
		double focusMeasure = 0;
		Mat lap = new Mat();
		if (src.channels() >= 3) {
			// Laplacian does not work when source and destination the same
			Mat gray = new Mat();
			Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
			Imgproc.Laplacian(gray, lap, CvType.CV_64F);
			gray.release();
		} else {
			Imgproc.Laplacian(src, lap, CvType.CV_64F);
		}
		MatOfDouble mean = new MatOfDouble();
		MatOfDouble std = new MatOfDouble();
		Core.meanStdDev(lap, mean, std);
		focusMeasure = std.get(0, 0)[0] * std.get(0, 0)[0];
		lap.release();
		mean.release();
		std.release();
		return focusMeasure;
	}

	/**
	 * OpenCV port of 'TENG' algorithm (Krotkov86)
	 * https://stackoverflow.com/a/7768918
	 * 
	 * @param src
	 *            The source image
	 * @param ksize
	 *            size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
	 * @return tenengrad
	 */
	public static double tenengrad(Mat src, int ksize) {
		Mat gray = new Mat();
		if (src.channels() >= 3) {
			Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
		} else {
			src.copyTo(gray);
		}
		double focusMeasure = 0;
		Mat Gx = new Mat();
		Mat Gy = new Mat();
		Imgproc.Sobel(gray, Gx, CvType.CV_64F, 1, 0, ksize, 1.0, 0.0);
		Imgproc.Sobel(gray, Gy, CvType.CV_64F, 0, 1, ksize, 1.0, 0.0);
		Mat FM = new Mat();
		Core.add(Gx.mul(Gx), Gy.mul(Gy), FM);
		focusMeasure = Core.mean(FM).val[0];
		Gx.release();
		Gy.release();
		FM.release();
		gray.release();
		return focusMeasure;
	}

	/**
	 * OpenCV port of 'GLVN' algorithm (Santos97)
	 * https://stackoverflow.com/a/7768918
	 * 
	 * @param src
	 *            The source image
	 * @return normalizedGraylevelVariance
	 */
	public static double normalizedGraylevelVariance(Mat src) {
		Mat gray = new Mat();
		if (src.channels() >= 3) {
			Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
		} else {
			src.copyTo(gray);
		}
		double focusMeasure = 0;
		MatOfDouble mean = new MatOfDouble();
		MatOfDouble std = new MatOfDouble();
		Core.meanStdDev(gray, mean, std);
		focusMeasure = (std.get(0, 0)[0] * std.get(0, 0)[0]) / mean.get(0, 0)[0];
		mean.release();
		std.release();
		gray.release();
		return focusMeasure;
	}

	/**
	 * Find average brightness across image. Is this root - mean - square or not?
	 * https://www.zohodiscussions.com/processing/topic/calculate-image-contrast-using-root-mean-square-rms
	 * https://github.com/jeffThompson/ProcessingTeachingSketches/blob/master/ImageProcessingAndOpenCV/MeasureImageBrightnessAndContrast/MeasureImageBrightnessAndContrast.pde
	 * 
	 * @param src
	 * @return
	 */
	public static double brightness(Mat src, boolean normalize) {
		double brigthness = 0;
		if (src.channels() >= 3) {
			double[] rgb = new double[3];
			for (int x = 0; x < src.rows(); x++) {
				for (int y = 0; y < src.cols(); y++) {
					src.get(x, y, rgb);
					brigthness += (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]); // scales RGB to perceived brightness
					if (normalize) {
						brigthness = brigthness / 255.0; // normalize to 0-1
					}
				}
			}
			brigthness = brigthness / (src.rows() * src.cols());

		} else {

		}
		LOGGER.log(Level.INFO, "Brightness: {0}", brigthness);
		return brigthness;
	}

	/**
	 * find contrast by comparing average brightness with current value
	 * 
	 * @param src
	 * @param brightness
	 * @param normalize
	 * @return
	 */
	public static double contrast(Mat src, double brightness, boolean normalize) {
		double contrast = 0;
		double pxIntensity = 0;
		if (src.channels() >= 3) {
			double[] rgb = new double[3];
			for (int x = 0; x < src.rows(); x++) {
				for (int y = 0; y < src.cols(); y++) {
					src.get(x, y, rgb);
					pxIntensity = (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]);
					if (normalize) {
						pxIntensity = pxIntensity / 255.0; // normalize to 0-1
					}
					contrast += Math.pow((brightness - pxIntensity), 2);
				}
			}

		} else {

		}
		LOGGER.log(Level.INFO, "Contrast: {0}", contrast);
		return contrast;
	}

	/**
	 * Sets the debugging windows and waits for keypress
	 */
	private static void SetDebugWindows() {
		HighGui.moveWindow("Canny", 0, 0);
		HighGui.moveWindow("Contours", (int) (EDGESIZE.width + 3 * BORDER_SIZE), 0);
		HighGui.moveWindow("Approx", 0, (int) (EDGESIZE.height + 3 * BORDER_SIZE));
		HighGui.moveWindow("Cropped", (int) (EDGESIZE.width + 3 * BORDER_SIZE), (int) (EDGESIZE.height + 3 * BORDER_SIZE));
		HighGui.resizeWindow("Canny", (int) (EDGESIZE.width + BORDER_SIZE), (int) (EDGESIZE.height + BORDER_SIZE));
		HighGui.resizeWindow("Contours", (int) (EDGESIZE.width + BORDER_SIZE), (int) (EDGESIZE.height + BORDER_SIZE));
		HighGui.resizeWindow("Approx", (int) (EDGESIZE.width + BORDER_SIZE), (int) (EDGESIZE.height + BORDER_SIZE));
		HighGui.resizeWindow("Cropped", (int) (EDGESIZE.width + BORDER_SIZE), (int) (EDGESIZE.height + BORDER_SIZE));
		HighGui.waitKey();
	}
}
