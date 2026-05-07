package preprocess;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.opencv.core.Mat;

public class TensorWriter
{
	public static void saveTensor(Mat tensor, String filePath) throws IOException
	{
		int height = tensor.rows();
		int width = tensor.cols();
		int channels = tensor.channels();

		float[] data = new float[height * width * channels];
		tensor.get(0, 0, data);

		try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath))))
		{
			dos.writeInt(width);
			dos.writeInt(height);
			dos.writeInt(channels);

			for (float v : data)
			{
				dos.writeFloat(v);
			}
		}
	}
}
