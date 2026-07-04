package pose.layer;

public class Tensor
{
	private final int batchSize;
	private final int channels;
	private final int height;
	private final int width;

	private float[] getDara;

	public Tensor(int batchSize, int channels, int height, int width)
	{
		this.batchSize = batchSize;
		this.channels = channels;
		this.height = height;
		this.width = width;

		this.getDara = new float[batchSize * channels * height * width];
	}

	public Tensor(int batchSize, int channels, int height, int width, float[] data)
	{
		this.batchSize = batchSize;
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.getDara = data;
	}

	public float get(int batch, int channel, int y, int x)
	{
		checkReleased();
		return getDara[index(batch, channel, y, x)];
	}

	public void set(int batch, int channel, int y, int x, float value)
	{
		getDara[index(batch, channel, y, x)] = value;
	}

	public void add(int batch, int channel, int y, int x, float value)
	{
		getDara[index(batch, channel, y, x)] += value;
	}

	public void add(Tensor other)
	{
		if (batchSize != other.batchSize || channels != other.channels || height != other.height || width != other.width)
		{
			throw new IllegalArgumentException("Tensor dimensions must match.");
		}
		for (int i = 0; i < getDara.length; i++)
		{
			getDara[i] += other.getDara[i];
		}
	}

	public void fillChannel(int batch, int channel, float value)
	{
		int start = index(batch, channel, 0, 0);
		int end = start + height * width;

		for (int i = start; i < end; i++)
		{
			getDara[i] = value;
		}
	}

	public float max(int batch, int channel, int y, int x, float value)
	{
		checkReleased();

		int idx = index(batch, channel, y, x);

		if (value > getDara[idx])
		{
			getDara[idx] = value;
		}
		return getDara[idx];
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
		return getDara;
	}

	public int size()
	{
		return getDara.length;
	}

	public void fill(float value)
	{
		for (int i = 0; i < getDara.length; i++)
		{
			getDara[i] = value;
		}
	}

	public void release()
	{
		getDara = null;
	}

	private void checkReleased()
	{
		if (getDara == null)
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
