package pose.layer;

public class UpSampleLayer extends Layer
{
	private final int scale;

	public UpSampleLayer(int scale)
	{
		this.scale = scale;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// Nearest Neighbors
		this.input = input;
		int batchSize = input.getBatchSize();
		int channels = input.getChannels();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = inHeight * scale;
		int outWidth = inWidth * scale;

		Tensor output = new Tensor(batchSize, channels, outHeight, outWidth);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int channel = 0; channel < channels; channel++)
			{
				for (int inY = 0; inY < inHeight; inY++)
				{
					for (int inX = 0; inX < inWidth; inX++)
					{
						int outY = inY * scale;
						int outX = inX * scale;
						float value = input.get(batch, channel, inY, inX);

						for (int i = 0; i < scale; i++)
						{
							for (int h = 0; h < scale; h++)
							{
								output.set(batch, channel, outY + i, outX + h, value);
							}
						}
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		int batchSize = gradient.getBatchSize();
		int channels = gradient.getChannels();
		int outHeight = gradient.getHeight() / scale;
		int outWidth = gradient.getWidth() / scale;

		Tensor gradOutput = new Tensor(batchSize, channels, outHeight, outWidth);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int channel = 0; channel < channels; channel++)
			{
				for (int y = 0; y < outHeight; y++)
				{
					for (int x = 0; x < outWidth; x++)
					{
						float sum = 0f;
						int startY = y * scale;
						int startX = x * scale;

						for (int dy = 0; dy < scale; dy++)
						{
							for (int dx = 0; dx < scale; dx++)
							{
								sum += gradient.get(batch, channel, startY + dy, startX + dx);
							}
						}
						gradOutput.set(batch, channel, y, x, sum);
					}
				}
			}
		}
		return gradOutput;
	}
}
