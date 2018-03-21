import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OcrPreProcessing {
	
	private static final Logger LOGGER = Logger.getLogger( OcrPreProcessing.class.getName() );

	public static Mat prepare(Mat source) {
		LOGGER.log(Level.INFO, "Prepare image for OCR");
		Mat output = new Mat(source.rows(), source.cols(), CvType.CV_8U);
		
		// Step: Gray Scale
		LOGGER.log(Level.FINE, "GrayScale");
		Mat gray = new Mat(source.rows(), source.cols(), CvType.CV_8U);
		Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
		gray.copyTo(output);
		
		// Step: Equalize Histogram after gray scaling
		LOGGER.log(Level.FINE, "Equalize Histogram after gray scaling");
		Imgproc.equalizeHist(gray, gray);
	
		// Step: BilateralFilter blurring for better keeping edges than GaussianBlur
		LOGGER.log(Level.FINE, "BilateralFilter Blur");
		Mat filterSource = new Mat(source.rows(), source.cols(), CvType.CV_8U);
		output.copyTo(filterSource);
		//[TODO] What are the best parameters for d and sigmaColor and sigmaSpace? Trial and error...
		//Imgproc.bilateralFilter(filterSource, output, 9, 300, 300);
		Imgproc.bilateralFilter(filterSource, output, 7, 75, 75);
		filterSource.release();
		
		// Step: Adaptive Threshold
		LOGGER.log(Level.FINE, "Adaptive Threshold");
		// [TODO] What are good values for blockSize and c? Trial and error...
		//Imgproc.adaptiveThreshold(output, output, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 4);
		//Imgproc.adaptiveThreshold(output, output, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
		Imgproc.adaptiveThreshold(output, output, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 13, 4);

		// Step: Erode to make the lines bigger
		LOGGER.log(Level.FINE, "Erode to make the threshold lines bigger");
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(3, 3));
        Imgproc.erode(output, output, element);
        element.release();

		// Step: Blend both images together
		LOGGER.log(Level.FINE, "Blending both images together");
		Core.addWeighted(output, 0.6, gray, 0.4, 0.0, output);
		gray.release();
		
		return output;
	}
}
