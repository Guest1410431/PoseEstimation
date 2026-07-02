package pose.layer;

public class Tensor
{
	private final int batchSize;
	private final int channels;
	private final int height;
	private final int width;

	private float[] data;

	public Tensor(int batchSize, int channels, int height, int width)
	{
		this.batchSize = batchSize;
		this.channels = channels;
		this.height = height;
		this.width = width;

		this.data = new float[batchSize * channels * height * width];
	}

	public Tensor(int batchSize, int channels, int height, int width, float[] data)
	{
		this.batchSize = batchSize;
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.data = data;
	}

	public float get(int batch, int channel, int y, int x)
	{
		checkReleased();
		return data[index(batch, channel, y, x)];
	}

	public void set(int batch, int channel, int y, int x, float value)
	{
		data[index(batch, channel, y, x)] = value;
	}

	public void add(int batch, int channel, int y, int x, float value)
	{
		data[index(batch, channel, y, x)] += value;
	}

	public void add(Tensor other)
	{
		if (batchSize != other.batchSize || channels != other.channels || height != other.height || width != other.width)
		{
			throw new IllegalArgumentException("Tensor dimensions must match.");
		}
		for (int i = 0; i < data.length; i++)
		{
			data[i] += other.data[i];
		}
	}

	public void fillChannel(int batch, int channel, float value)
	{
		int start = index(batch, channel, 0, 0);
		int end = start + height * width;

		for (int i = start; i < end; i++)
		{
			data[i] = value;
		}
	}

	public float max(int batch, int channel, int y, int x, float value)
	{
		checkReleased();

		int idx = index(batch, channel, y, x);

		if (value > data[idx])
		{
			data[idx] = value;
		}
		return data[idx];
	}

	public boolean inBounds(int y, int x)
	{
		return y >= 0 && y < height && x >= 0 && x < width;
	}

	public int index(int batch, int channel, int y, int x)
	{
		return ((batch * channels + channel) * height + y) * width + x;
	}

	public int getBatchSize()
	{
		return batchSize;
	}

	public int getChannels()
	{
		return channels;
	}

	public int getHeight()
	{
		return height;
	}

	public int getWidth()
	{
		return width;
	}

	public float[] getData()
	{
		return data;
	}

	public int size()
	{
		return data.length;
	}

	public void fill(float value)
	{
		for (int i = 0; i < data.length; i++)
		{
			data[i] = value;
		}
	}

	public void release()
	{
		data = null;
	}

	private void checkReleased()
	{
		if (data == null)
		{
			throw new IllegalStateException("Tensor has been released.");
		}
	}

	@Override
	public String toString()
	{
		return "Tensor[" + batchSize + ", " + channels + ", " + height + ", " + width + "]";
	}
}
