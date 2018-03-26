import java.util.Comparator;
import org.opencv.core.Point;

/**
 * @author dansailer 
 * Compare two points based on the sum of their coordinates.
 * In a quadrilateral, top left corner has smallest x+y sum and bottom
 * right corner has largest x+y sum.
 */
public class CoordinateSumComparator implements Comparator<Point> {
	@Override
	public int compare(Point lhs, Point rhs) {
		return Double.valueOf(lhs.x + lhs.y).compareTo(rhs.x + rhs.y);
	}
}
