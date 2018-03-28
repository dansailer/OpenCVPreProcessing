import java.io.File;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class OpenCVPreProcessing {

	private static final Logger LOGGER = Logger.getLogger(OpenCVPreProcessing.class.getName());
	/**
	 * Enable debug image output
	 */
	public static boolean DEBUG = Boolean.parseBoolean(System.getProperty("DEBUG", "false"));

	public static void main(String[] args) {

		// [TODO]
		// - verify croped image has minimal size (1/4 of the original?)
		// 5a8aa3540cf2a5b482619260.jpg, 5a8aa3560cf2a5b482619267.jpg
		// 5a8c57080cf2a5b4826195b5.jpg
		// - verify that the image is not pure white. 5a8aa4ac0cf2a5b4826192a8.jpg
		// - don't change small images: 5a8eca890cf2a5b482619b68.jpg
		// - verify minimal filesize for reading...
		// - improve readability 5a8f457f0cf2a5b482619d72.jpg --> Disable Erode

		// OKP rechnung wieder an die OKP? 5a8eca890cf2a5b482619b68.jpg
		
		// Plain awesome!
		// - 5a81e4c00cf29fa5dabe5e20.jpg
		// - 5aa7fcaa0cf2196ddadb3ee7.jpg

		// load the OpenCV native library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		LOGGER.setLevel(Level.SEVERE);
		OcrPreProcessing.LOGGER.setLevel(Level.SEVERE);

		if (args.length == 0) {
			LOGGER.log(Level.SEVERE, "No images given to process!");
			return;
		}
		for (String arg : args) {
			File f = new File(arg);
			if (!f.exists() || !f.isFile()) {
				LOGGER.log(Level.SEVERE, "Argument {0} is not a file!", arg);
				return;
			}
			LOGGER.log(Level.INFO, "Processing image {0} ...", arg);
			Mat source = Imgcodecs.imread(arg, Imgproc.COLOR_BGR2RGB);
			ArrayList<Point> corners = OcrPreProcessing.detectPage(source, OcrPreProcessing.MIN_PAGE_FRACTION);
			if (DEBUG) {
				corners = OcrPreProcessing.detectPageWithMinRect(source, OcrPreProcessing.MIN_PAGE_FRACTION);
				corners = OcrPreProcessing.detectPageConvexHull(source, OcrPreProcessing.MIN_PAGE_FRACTION);
			}
			Mat cropped = OcrPreProcessing.transform(source, corners);
			Mat ocr = OcrPreProcessing.prepare(cropped, false, true);
			double blurSrc = OcrPreProcessing.varianceOfLaplacian(source);
			double modifiedSrc = OcrPreProcessing.modifiedLaplacian(source);
			if( blurSrc < 70 && modifiedSrc < 4) {
				LOGGER.log(Level.SEVERE, "\n BLUR - Source: varianceOfLaplacian: {1}, modifiedLaplacian: {2}, file: {0}", new Object[] {arg, blurSrc, modifiedSrc});
			}
			if (!DEBUG) {
				Imgcodecs.imwrite(arg.replace(".jpg", ".png"), ocr);
			}
			cropped.release();
			ocr.release();
			source.release();
		}
	}

}
