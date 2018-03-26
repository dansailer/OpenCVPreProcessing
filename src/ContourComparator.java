import java.util.Comparator;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;

/**
 * @author dansailer
 * Compare two OpenCV contours based on their area. Larger contour areas are coming first.
 */
public class ContourComparator implements Comparator<MatOfPoint> {
	@Override
	public int compare(MatOfPoint lhs, MatOfPoint rhs) {
		return Double.valueOf(Imgproc.contourArea(rhs)).compareTo(Imgproc.contourArea(lhs));
	}
}
