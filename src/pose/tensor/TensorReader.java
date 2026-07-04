package pose.tensor;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class TensorReader
{
	public static Tensor loadTensor(String filePath)
	{
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath))))
		{
			int width = dis.readInt();
			int height = dis.readInt();
			int channels = dis.readInt();

			Tensor tensor = Tensor.acquire(1, channels, height, width);

			float[] data = tensor.getData();

			for (int i = 0; i < data.length; i++)
			{
				data[i] = dis.readFloat();
			}
			return tensor;
		}
		catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		return null;
	}
}
