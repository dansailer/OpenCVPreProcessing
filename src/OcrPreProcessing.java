import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * @author dansailer
 *
 */
public class OcrPreProcessing {

	private static final Logger LOGGER = Logger.getLogger(OcrPreProcessing.class.getName());
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
			Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
			gray.copyTo(output);

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

			// Step: Erode to make the lines bigger
			LOGGER.log(Level.FINER, "Erode to make the threshold lines bigger");
			Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
			Imgproc.erode(output, output, element);
			element.release();

			// Step: Blend both images together
			if (blend) {
				LOGGER.log(Level.FINER, "Blending both images together");
				Core.addWeighted(output, 0.6, gray, 0.4, 0.0, output);
				gray.release();
			}
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "Preparing image for OCR failed with '{0}'", exc.getMessage());
			source.copyTo(output);
		}

		return output;
	}

	/*
	 * public static Mat crop(Mat source) { try { LOGGER.log(Level.INFO,
	 * "Cropping image"); Mat working = new Mat(source.rows(), source.cols(),
	 * source.type()); source.copyTo(working);
	 * 
	 * // Step: Gray Scale LOGGER.log(Level.FINE, "GrayScale");
	 * Imgproc.cvtColor(source, working, Imgproc.COLOR_BGR2GRAY);
	 * 
	 * // Step: BilateralFilter blurring for better keeping edges than GaussianBlur
	 * LOGGER.log(Level.FINE, "BilateralFilter Blur"); Mat filterSource = new
	 * Mat(working.rows(), working.cols(), CvType.CV_8U);
	 * working.copyTo(filterSource); Imgproc.bilateralFilter(filterSource, working,
	 * 7, 75, 75); filterSource.release();
	 * 
	 * // Step: Get ratio and size for better edge detection LOGGER.log(Level.FINE,
	 * "Resizing..."); double ratio = Math.min(EDGESIZE.width / source.width(),
	 * EDGESIZE.height / source.height()); Size newSize = new Size(source.width() *
	 * ratio, source.height() * ratio); Imgproc.resize(working, working, newSize);
	 * 
	 * // Step: Add black border in case page is clipped
	 * Core.copyMakeBorder(working, working, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
	 * BORDER_SIZE, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
	 * 
	 * // Step: GaussianBlur LOGGER.log(Level.FINE, "GaussianBlur");
	 * Imgproc.GaussianBlur(working, working, new Size(5, 5), 0.0);
	 * 
	 * // Step: Canny LOGGER.log(Level.FINE, "Canny"); Mat edges = new
	 * Mat(working.rows(), working.cols(), working.type()); // [TODO] Calculate auto
	 * thresholds //
	 * https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-
	 * detection-with-python-and-opencv/ Imgproc.Canny(working, edges, 75, 200, 3,
	 * true);
	 * 
	 * // Step: Find contours, sorte and take the largest couple
	 * LOGGER.log(Level.FINE, "Find contours, sorte and take the largest couple");
	 * ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>(); Mat hierarchy =
	 * new Mat(); Imgproc.findContours(edges, contours, hierarchy,
	 * Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE); hierarchy.release(); int
	 * largeNumber = Math.min(contours.size(), NUMBER_OF_LARGE_CONTOURS);
	 * LOGGER.log(Level.FINER, "Number of contours: {0}", contours.size());
	 * contours.sort(new ContourComparator()); List<MatOfPoint> largeContours =
	 * contours.subList(0, largeNumber);
	 * 
	 * // Step: Loop over and find 4 cornered approximated convex contour
	 * LOGGER.log(Level.FINE, "Loop over and find 4 cornered convex contour");
	 * MatOfPoint2f pageContour = new MatOfPoint2f(); for (MatOfPoint cnt :
	 * largeContours) { // approximate the contour MatOfPoint2f c2f = new
	 * MatOfPoint2f(cnt.toArray()); double epsilon = Imgproc.arcLength(c2f, true);
	 * MatOfPoint2f approx2f = new MatOfPoint2f(); MatOfPoint approx = new
	 * MatOfPoint(); Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
	 * approx2f.convertTo(approx, CvType.CV_32S); boolean convex =
	 * Imgproc.isContourConvex(approx); LOGGER.log(Level.FINER,
	 * "Approximated contour - Total: {0} ElemSize: {1} Convex: {2} Continious: {3}"
	 * , new Object[] { approx2f.total(), approx2f.elemSize(), convex,
	 * approx2f.isContinuous() }); // approximated contour has four points and is
	 * convex --> page found if (approx2f.total() == 4 && convex) { pageContour =
	 * approx2f; break; } }
	 * 
	 * // Step: If page found de warp and crop if (!pageContour.empty()) { Point[]
	 * corners = pageContour.toArray(); ArrayList<Point> scaledCorners = new
	 * ArrayList<Point>(); MatOfPoint contour = new MatOfPoint(); for (Point p :
	 * corners) { scaledCorners.add(new Point(Math.round((p.x - BORDER_SIZE) /
	 * ratio), Math.round((p.y - BORDER_SIZE) / ratio))); }
	 * contour.fromList(scaledCorners); working = transform(source, scaledCorners);
	 * return working; } } catch (Exception exc) {
	 * 
	 * } return source; }
	 */

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
	 *            The image to detect the page in
	 * @return The four corners of the page
	 */
	public static ArrayList<Point> detectPage(Mat source) {
		LOGGER.log(Level.FINE, "Detect paper page borders");
		ArrayList<Point> scaledCorners = new ArrayList<Point>();
		MatOfPoint2f pageContour = new MatOfPoint2f();
		double ratio = 1.0;

		try {
			Mat working = new Mat(source.rows(), source.cols(), source.type());
			source.copyTo(working);

			// Step: Gray Scale
			LOGGER.log(Level.FINER, "GrayScale");
			Imgproc.cvtColor(source, working, Imgproc.COLOR_BGR2GRAY);

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

			// Step: Canny
			LOGGER.log(Level.FINER, "Canny edge detection");
			Mat edges = new Mat(working.rows(), working.cols(), working.type());
			// [TODO] Calculate auto thresholds
			// https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
			Imgproc.Canny(working, edges, 75, 200, 3, true);

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

			// Step: Loop over and find 4 cornered approximated convex contour searching
			// from large to smaller
			LOGGER.log(Level.FINER, "Loop over and find 4 cornered convex contour searching from large to smaller");
			for (MatOfPoint cnt : largeContours) {
				// approximate the contour
				MatOfPoint2f c2f = new MatOfPoint2f(cnt.toArray());
				double epsilon = Imgproc.arcLength(c2f, true);
				MatOfPoint2f approx2f = new MatOfPoint2f();
				MatOfPoint approx = new MatOfPoint();
				Imgproc.approxPolyDP(c2f, approx2f, 0.02 * epsilon, true);
				approx2f.convertTo(approx, CvType.CV_32S);
				boolean convex = Imgproc.isContourConvex(approx);
				LOGGER.log(Level.FINER, "Approximated contour - Total: {0} Convex: {1} Continious: {2}",
						new Object[] { approx2f.total(), convex, approx2f.isContinuous() });
				// approximated contour has four points and is convex --> page found
				if (approx2f.total() == 4 && convex) {
					pageContour = approx2f;
					break;
				}
			}
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "Finding page failed with '{0}'", exc.getMessage());
		}

		// Step: If page found, scale and calculate corners
		LOGGER.log(Level.FINER, "If page found, scale and calculate corners");
		if (!pageContour.empty()) {
			Point[] corners = pageContour.toArray();
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
			LOGGER.log(Level.SEVERE, "Transformation failed with '{0}'", exc.getMessage());
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
}
