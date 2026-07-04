package pose.tensor;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class TensorWriter
{
	public static void saveTensor(Tensor tensor, String filePath) throws IOException
	{
		int height = tensor.getHeight();
		int width = tensor.getWidth();
		int channels = tensor.getChannels();

		float[] data = tensor.getData();

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
