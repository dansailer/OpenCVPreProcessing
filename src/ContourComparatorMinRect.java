import java.util.Comparator;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

/**
 * @author dansailer Compare two OpenCV contours based on their area. Larger
 *         contour areas are coming first. Instead of using the OpenCV function
 *         contourArea, the comparator uses the size of the minAreaRect
 *         function, as the contourArea function does not work if the contour is
 *         not closed.
 */
public class ContourComparatorMinRect implements Comparator<MatOfPoint> {

	private static final Logger LOGGER = Logger.getLogger(ContourComparatorMinRect.class.getName());

	@Override
	public int compare(MatOfPoint lhs, MatOfPoint rhs) {
		double lhsArea = calculateMinRectArea(lhs);
		double rhsArea = calculateMinRectArea(rhs);
		return Double.valueOf(rhsArea).compareTo(lhsArea);
	}

	/**
	 * Calculating the area of the bounding minimum area rectangle
	 * 
	 * @param mop
	 *            contour to get a area size
	 * @return bounding minimum area rectangle size
	 */
	public static double calculateMinRectArea(MatOfPoint mop) {
		double result = 0;
		try {
			MatOfPoint2f mop2f = new MatOfPoint2f(mop.toArray());
			RotatedRect area = Imgproc.minAreaRect(mop2f);
			result = area.size.width * area.size.height;
		} catch (Exception exc) {
			LOGGER.log(Level.SEVERE, "calculateArea failed with \"{0}\"", exc.getMessage());
		}
		return result;
	}
}
