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
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import pose.layer.Tensor;

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

	private Tensor guassianKernal;

	public ImagePreprocessing()
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		readAnnotations();

		System.out.println("Loaded annotations for: " + rawImages.size() + " images");

		guassianKernal = createGuassianKernal();

		processAllImages();
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

	private void processAllImages()
	{
		int processed = 0;
		int total = rawImages.size();

		for (String file : rawImages.keySet())
		{
			processSingleImage(file);
			
			if (++processed % 500 == 0)
			{
				System.out.println("Processed: " + processed + "/" + total);
			}
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
			Tensor imageTensor = imageToTensor(input);
			saveTensor(imageTensor, "res/training_data/tensorImages/" + fileName, personId);
			
			List<Annotation> joints = scaledPeople.get(personId);

			Tensor heatmaps = generateHeatmaps(joints);

			saveTensor(heatmaps, "res/training_data/heatmaps/" + fileName, personId);

			heatmaps.release();
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

	private Tensor createGuassianKernal()
	{
		Tensor kernal = new Tensor(1, 1, GUASSIAN_BLUR, GUASSIAN_BLUR);
		int center = GUASSIAN_BLUR / 2;
		double twoSigmaSquared = 2 * SIGMA * SIGMA;

		float max = 0;

		for (int y = 0; y < GUASSIAN_BLUR; y++)
		{
			for (int x = 0; x < GUASSIAN_BLUR; x++)
			{
				float value = (float) Math.exp(-((x - center) * (x - center) + (y - center) * (y - center)) / twoSigmaSquared);

				kernal.set(0, 0, y, x, value);

				if (value > max)
				{
					max = value;
				}
			}
		}
		// Normalize to [0,1]
		for (int y = 0; y < GUASSIAN_BLUR; y++)
		{
			for (int x = 0; x < GUASSIAN_BLUR; x++)
			{
				kernal.set(0, 0, y, x, kernal.get(0, 0, y, x) / max);
			}
		}
		return kernal;
	}

	private Tensor generateHeatmaps(List<Annotation> joints)
	{
		final int NUM_JOINTS = 16;
		int jointId = 0;

		Tensor heatmaps = new Tensor(1, NUM_JOINTS, (int) RESIZE.height, (int) RESIZE.width);

		for (Annotation joint : joints)
		{
			generateHeatmap(heatmaps, jointId++, joint.getxPos(), joint.getyPos());
		}
		return heatmaps;
	}

	private void generateHeatmap(Tensor heatmaps, int channel, int xPos, int yPos)
	{
		int half = GUASSIAN_BLUR / 2;

		for (int ky = 0; ky < GUASSIAN_BLUR; ky++)
		{
			for (int kx = 0; kx < GUASSIAN_BLUR; kx++)
			{
				int x = xPos + kx - half;
				int y = yPos + ky - half;

				if (x < 0 || x >= heatmaps.getWidth() || y < 0 || y >= heatmaps.getHeight())
				{
					continue;
				}
				float kernelValue = guassianKernal.get(0, 0, ky, kx);

				float current = heatmaps.get(0, channel, y, x);

				if (kernelValue > current)
				{
					heatmaps.set(0, channel, y, x, kernelValue);
				}
			}
		}
	}

	private Tensor imageToTensor(Mat image)
	{
		Tensor tensor = new Tensor(1, 3, image.rows(), image.cols());

		double[] pixel = new double[3];

		for (int y = 0; y < image.rows(); y++)
		{
			for (int x = 0; x < image.cols(); x++)
			{
				image.get(y, x, pixel);

				tensor.set(0, 0, y, x, (float) (pixel[2] / 255.0)); // R
				tensor.set(0, 1, y, x, (float) (pixel[1] / 255.0)); // G
				tensor.set(0, 2, y, x, (float) (pixel[0] / 255.0)); // B
			}
		}
		return tensor;
	}

	private void saveTensor(Tensor tensor, String path, int personId)
	{
		try
		{
			path = path.substring(0, path.length() - 4);

			TensorWriter.saveTensor(tensor, path + "_person" + personId + ".tensor");
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
