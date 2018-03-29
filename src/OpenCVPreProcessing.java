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
		
		/*
		 * Issue:
		 * - 5a81e4c00cf29fa5dabe5e20.jpg
		 * 
		 * Blurry:
		 * - 5a96f21a0cf21002233454d8.jpg
		 * - 5a96f2070cf21002233454d6.jpg
		 * - 5aa7fcaa0cf2196ddadb3ee7.jpg (not detected)
		 * 
		 * Page cut improvement examples:
		 * - 5a9462ac0cf2100223344a71.jpg
		 * - 5a9660760cf2100223345223.jpg
		 * - 5aa158ea0cf2196ddadb278a.jpg
		 * - 5aa5714d0cf2196ddadb3038.jpg
		 * 
		 * Page too small:
		 * - 5aa108b10cf2196ddadb2717.jpg (640x480)
		 * - 5aa108400cf2196ddadb2715.jpg (640x480 but would be ok....)
		 * - 5ab4d3b70cf2196ddadb6097.jpg (462x640)
		 * 
		 * Page too dark:
		 * - 5aaffffd0cf2196ddadb5123.jpg
		 */
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
			if (!f.exists() || !f.isFile() || arg == "") {
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
//			LOGGER.log(Level.SEVERE, "\n BLUR - Source: varianceOfLaplacian: {1}, modifiedLaplacian: {2}, file: {0}", new Object[] {arg, blurSrc, modifiedSrc});
			
			if (!DEBUG) {
				Imgcodecs.imwrite(arg.replace(".jpg", ".png"), ocr);
			}
			cropped.release();
			ocr.release();
			source.release();
		}
	}

}
