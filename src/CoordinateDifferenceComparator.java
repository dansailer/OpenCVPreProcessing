import java.util.Comparator;
import org.opencv.core.Point;

/**
 * @author dansailer 
 * Compare two points based on the difference of their coordinates.
 * In a quadrilateral, top right corner has smallest x-y difference and bottom
 * left corner has largest x-y difference.
 */
public class CoordinateDifferenceComparator implements Comparator<Point> {
	@Override
	public int compare(Point lhs, Point rhs) {
		return Double.valueOf(rhs.x - rhs.y).compareTo(lhs.x - lhs.y);
	}
}
