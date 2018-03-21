import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class OpenCVPreProcessing {

	private static final Logger LOGGER = Logger.getLogger( OpenCVPreProcessing.class.getName() );
	
	public static void main(String[] args) {
		// load the OpenCV native library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		LOGGER.setLevel(Level.FINE);
		
		if (args.length == 0) {
			LOGGER.log(Level.SEVERE, "No images given to process!");
			return;
		}
		for(String arg : args) {
			File f = new File(arg);
			if(!f.exists() || !f.isFile()) {
				LOGGER.log(Level.SEVERE, "Argument {0} is not a file!", arg);
				return;
			}
			LOGGER.log(Level.INFO, "Processing image {0} ...", arg);
			Mat source = Imgcodecs.imread(arg, Imgproc.COLOR_BGR2RGB);
			Mat output = OcrPreProcessing.prepare(source);
			Imgcodecs.imwrite(arg.replace(".", "_output."), output);
			output.release();
			source.release();
		}
	}

}
