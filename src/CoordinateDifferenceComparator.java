import java.util.Comparator;
import org.opencv.core.Point;

public class CoordinateDifferenceComparator implements Comparator<Point> {
	@Override
    public int compare(Point lhs, Point rhs) {
        return Double.valueOf(rhs.x-rhs.y).compareTo(lhs.x-lhs.y);
    }
}
