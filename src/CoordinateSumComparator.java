import java.util.Comparator;
import org.opencv.core.Point;

public class CoordinateSumComparator implements Comparator<Point> {
	@Override
    public int compare(Point lhs, Point rhs) {
        return Double.valueOf(lhs.x+lhs.y).compareTo(rhs.x+rhs.y);
    }
}
