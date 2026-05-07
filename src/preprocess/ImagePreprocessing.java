package preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessing
{
	private final int IMAGE_WIDTH = 1280;
	private final int IMAGE_HEIGHT = 720;
	private final Size RESIZE = new Size(256, 144);
	private final double RESIZE_WIDTH_FACTOR = RESIZE.width / IMAGE_WIDTH;
	private final double RESIZE_HEIGHT_FACTOR = RESIZE.height / IMAGE_HEIGHT;
	private final int GUASSIAN_BLUR = 20;
	private final float SIGMA = 2.5f;

	private final String ANNOTATION_FILE_NAME = "mpii_annotations.csv";
	private final String COMMA_DELIMITER = ",";

	private Map<String, Map<Integer, List<Annotation>>> rawImages;

	private Mat guassianKernal;

	public ImagePreprocessing()
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		readAnnotations();

		System.out.println("Loaded annotations for: " + rawImages.size() + " images");

		guassianKernal = createGuassianKernal();

		processAllImagesSafely();
		System.out.println("Done");
	}

	private void readAnnotations()
	{
		rawImages = new HashMap<>();

		try (BufferedReader br = new BufferedReader(new FileReader("res/training_data/annotations/" + ANNOTATION_FILE_NAME)))
		{
			String line;

			while ((line = br.readLine()) != null)
			{
				String[] values = line.split(COMMA_DELIMITER);
				String imageFile = values[0];
				int personId = Integer.parseInt(values[1]);
				int jointId = Integer.parseInt(values[2]);
				int xPos = Math.round(Float.parseFloat(values[3]));
				int yPos = Math.round(Float.parseFloat(values[4]));
				int visibility = Integer.parseInt(values[5]);

				rawImages.computeIfAbsent(imageFile, k -> new HashMap<>()).computeIfAbsent(personId, k -> new ArrayList<>()).add(new Annotation(jointId, xPos, yPos, visibility));
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	private void processAllImagesSafely()
	{
		int processed = 0;
		int total = rawImages.size();

		for (String file : rawImages.keySet())
		{
			processSingleImage(file);

			System.out.println("Processed: " + ++processed + "/" + total);

			System.gc();
		}
	}

	private void processSingleImage(String fileName)
	{
		Mat input = Imgcodecs.imread("res/training_data/images/" + fileName);

		if (input == null || input.empty())
		{
			System.out.println("Skipping invalid image: " + fileName);
			return;
		}
		Mat resized = new Mat();
		Imgproc.resize(input, resized, RESIZE);
		input.release();

		Map<Integer, List<Annotation>> originalPeople = rawImages.get(fileName);
		Map<Integer, List<Annotation>> scaledPeople = scaleAnnotations(originalPeople);

		for (Integer personId : scaledPeople.keySet())
		{
			List<Annotation> joints = scaledPeople.get(personId);

			List<Mat> heatmaps = generateHeatmaps(joints);

			Mat tensor = mergeToTensor(heatmaps);

			saveTensor(tensor, fileName, personId);

			for (Mat h : heatmaps)
			{
				h.release();
			}
			tensor.release();
		}
		resized.release();
	}

	private Map<Integer, List<Annotation>> scaleAnnotations(Map<Integer, List<Annotation>> people)
	{
		Map<Integer, List<Annotation>> newPeople = new HashMap<>();

		for (Entry<Integer, List<Annotation>> entry : people.entrySet())
		{
			int personId = entry.getKey();
			List<Annotation> joints = entry.getValue();
			List<Annotation> newJoints = new ArrayList<>();

			for (Annotation a : joints)
			{
				Annotation scaled = new Annotation(a.getJointId(), (int) (a.getxPos() * RESIZE_WIDTH_FACTOR), (int) (a.getyPos() * RESIZE_HEIGHT_FACTOR), a.getVisibility());
				newJoints.add(scaled);
			}
			newPeople.put(personId, newJoints);
		}
		return newPeople;
	}

	private Mat createGuassianKernal()
	{
		Mat kernal = new Mat(GUASSIAN_BLUR, GUASSIAN_BLUR, CvType.CV_32FC1);
		int center = GUASSIAN_BLUR / 2;
		double twoSigmaSquared = 2 * SIGMA * SIGMA;

		for (int y = 0; y < GUASSIAN_BLUR; y++)
		{
			for (int x = 0; x < GUASSIAN_BLUR; x++)
			{
				double dx = x - center;
				double dy = y - center;

				double value = Math.exp(-((dx * dx) + (dy * dy)) / twoSigmaSquared);
				kernal.put(y, x, value);
			}
		}
		Core.normalize(kernal, kernal, 0, 1, Core.NORM_MINMAX);
		return kernal;
	}

	private Mat generateHeatmap(int xPos, int yPos)
	{
		Mat heatmap = Mat.zeros((int) RESIZE.height, (int) RESIZE.width, CvType.CV_32FC1);

		int half = GUASSIAN_BLUR / 2;

		int x1 = (int) Math.max(0, Math.floor(xPos - half));
		int y1 = (int) Math.max(0, Math.floor(yPos - half));
		int x2 = (int) Math.max(0, Math.ceil(xPos - half));
		int y2 = (int) Math.max(0, Math.ceil(yPos - half));

		int kx1 = half - (xPos - x1);
		int ky1 = half - (yPos - y1);
		int kx2 = kx1 + (x2 - x1);
		int ky2 = ky1 + (y2 - y1);
		
		if (x2 <= x1 || y2 <= y1) 
		{
		    return Mat.zeros(heatmap.rows(), heatmap.cols(), CvType.CV_32FC1);
		}
		Mat roiHeatmap = heatmap.submat(y1, y2, x1, x2);
		Mat roiKernal = guassianKernal.submat(ky1, ky2, kx1, kx2);

		Core.max(roiHeatmap, roiKernal, roiHeatmap);

		return heatmap;
	}

	public List<Mat> generateHeatmaps(List<Annotation> joints)
	{
		List<Mat> heatmaps = new ArrayList<>();

		for (Annotation ann : joints)
		{
			heatmaps.add(generateHeatmap(ann.getxPos(), ann.getyPos()));
		}
		return heatmaps;
	}

	private Mat mergeToTensor(List<Mat> heatmaps)
	{
		Mat tensor = new Mat();
		Core.merge(heatmaps, tensor);
		return tensor;
	}

	private void saveTensor(Mat tensor, String imageName, int personId)
	{
		try
		{
			imageName = imageName.substring(0, imageName.length()-4);
			
			TensorWriter.saveTensor(tensor, "res/training_data/heatmaps/" + imageName + "_person" + personId + ".tensor");
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
